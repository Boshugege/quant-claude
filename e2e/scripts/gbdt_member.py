# -*- coding: utf-8 -*-
"""树模型成员（10-24，用户选定的结构性方向）：在端到端自己的输入张量上训练梯度提升树，作为集成里的另一类模型。
特征只用 e04 数据集的四个张量（1 分钟窗口、日内窗口 token、日线 token、同伴），按通用方式池化（末值、短／长窗口均值、标准差），
不加任何人工因子公式。树用 torch 写的直方图梯度提升（逐层生长，特征按训练样本分位数分箱为 uint8），在 GPU（MPS／CUDA）上训练。
目标与 Transformer 相同：逐截面秩 → 正态分位数；09:30 截面不参与训练。
用法（fold-A 原型）：python e2e/scripts/gbdt_member.py <run 名> [--small] [--trees N] [--depth D] [--lr x]
输出 e2e/runs/<run>/val_pred.parquet（2023）、pred_2024.parquet，可直接给 cmp_runs.py／ens_pred.py。"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--small", action="store_true", help="只用 2022-01～02 训练、2023-01 验证，测试速度与正确性")
ap.add_argument("--trees", type=int, default=400)
ap.add_argument("--depth", type=int, default=6)
ap.add_argument("--lr", type=float, default=0.05)
ap.add_argument("--rows", type=float, default=0.3, help="每棵树的行抽样比例")
ap.add_argument("--cols", type=float, default=0.5, help="每棵树的特征抽样比例")
ap.add_argument("--min-leaf", type=int, default=2000)
ap.add_argument("--lam", type=float, default=1.0)
ap.add_argument("--bins", type=int, default=64)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fields", default="base", help="1 分钟字段集合（base／ob1／full）")
ap.add_argument("--norm", default="rel2", help="1 分钟输入变换（rel2／rel3：买卖同类字段共用参照）")
ap.add_argument("--win", default="shape,book,tod", help="树用的窗口通道组（端到端自己设计的 30 分钟通道）")
ap.add_argument("--aux-w", type=float, default=0.5, help="e16 目标里多周期辅助项的权重（树单独可调，越大越慢）")
ap.add_argument("--bags", type=int, default=1, help="同一份分箱数据上训练几个不同抽样种子的树模型，输出取平均（降噪、降换手）")
ap.add_argument("--tgt", default="plain", help="plain：逐截面秩→正态；e16：与 Transformer 同一训练目标（秩目标 + 0.5×之后 2 个截面 + 代理风格／伪行业残差）")
args = ap.parse_args()
OUT = os.path.join("e2e", "runs", args.run)
os.makedirs(OUT, exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
T.set_fields(args.fields)
T.E04_NORM = args.norm
T.WIN_GROUPS = args.win
T.E04_CFG["n_win_feat"] = T.n_win_feat()
T.E04_CFG["fuse"] = True
fb = FeatherBackend()
log = open(os.path.join(OUT, "log.txt"), "a")


def say(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); log.write(s + "\n"); log.flush()


_ST = [None]


def flatten(X):
    """四个张量 → 平铺特征（float32）。只做通用池化。"""
    Xf, Xi, Xd, Xp = X
    view = os.environ.get("GB_VIEW", "")
    if view == "slow":                                                             # 10-27 慢信息树：只用日线 token（之前完整交易日）与同伴画像
        d = Xd.astype(np.float32)
        with np.errstate(all="ignore"):
            out = [d[:, -k, :] for k in (1, 2, 3, 4, 5)] + [np.nanmean(d[:, -5:, :], 1), np.nanmean(d, 1), np.nanstd(d, 1), Xp.astype(np.float32)]
        F = np.concatenate(out, 1); F[~np.isfinite(F)] = np.nan
        return F
    if view == "fine":                                                             # 10-27 盘口细节树：只用 1 分钟窗口（三档全字段 rel3），加逐档买卖不平衡与每笔规模不平衡
        m_, s_ = np.asarray(_ST[0][0], np.float32), np.asarray(_ST[0][1], np.float32)   # 数据集已标准化，先还原成 rel3 值再算比值
        f = Xf.astype(np.float32) * s_ + m_
        nm = list(T.FEATURE_COLS)
        def col(c):
            return np.expm1(f[:, :, nm.index(c)])                                  # rel3：同组字段共用参照，比值与参照无关
        out = [f[:, -k, :] for k in (1, 2, 3, 5)]
        with np.errstate(all="ignore"):
            out += [np.nanmean(f[:, -5:, :], 1), np.nanmean(f[:, -15:, :], 1), np.nanstd(f[:, -15:, :], 1)]
            ims = []
            for kind in ("volume", "num_orders"):
                bs = [col("bid_%s%d" % (kind, j)) for j in (1, 2, 3)]; as_ = [col("ask_%s%d" % (kind, j)) for j in (1, 2, 3)]
                for b, a in list(zip(bs, as_)) + [(bs[0] + bs[1] + bs[2], as_[0] + as_[1] + as_[2])]:
                    ims.append((b - a) / (b + a))
            for j in (1, 2, 3):                                                    # 每笔挂单规模的买卖对比：log(买量/买笔数) − log(卖量/卖笔数)
                ims.append(np.log((col("bid_volume%d" % j) + 1e-6) / (col("bid_num_orders%d" % j) + 1e-6)) - np.log((col("ask_volume%d" % j) + 1e-6) / (col("ask_num_orders%d" % j) + 1e-6)))
            I = np.stack(ims, 2)                                                   # (N, L, 11)
            out += [I[:, -1, :], I[:, -2, :], np.nanmean(I[:, -5:, :], 1), np.nanmean(I[:, -15:, :], 1), np.nanmean(I, 1), np.nanstd(I[:, -15:, :], 1)]
        F = np.concatenate(out, 1); F[~np.isfinite(F)] = np.nan
        return F
    out = []
    f = Xf.astype(np.float32)
    rich = os.environ.get("GB_RICH") == "1"                                       # G2：更细的时间分辨率（同样只是池化）
    with np.errstate(all="ignore"):
        if rich:
            out += [f[:, -k, :] for k in (1, 2, 3, 5, 10, 20, 30)] + [np.nanmean(f[:, -5:, :], 1), np.nanmean(f[:, -15:, :], 1), np.nanmean(f, 1), np.nanstd(f, 1)]
        else:
            out += [f[:, -1, :], np.nanmean(f[:, -5:, :], 1), np.nanmean(f[:, -15:, :], 1), np.nanmean(f, 1), np.nanstd(f, 1)]
        i = Xi.astype(np.float32)
        out += [i[:, -k, :] for k in ((1, 2, 3, 4, 5, 6, 7, 8) if rich else (1, 2, 3, 4))] + [np.nanmean(i, 1)]
        d = Xd.astype(np.float32)
        out += [d[:, -k, :] for k in ((1, 2, 3, 4, 5) if rich else (1, 2))] + [np.nanmean(d, 1)]
    out.append(Xp.astype(np.float32))
    F = np.concatenate(out, 1)
    F[~np.isfinite(F)] = np.nan
    return F


def make_edges(F, bins, seed):
    rs = np.random.RandomState(seed)
    s = F[rs.choice(len(F), min(len(F), 400000), replace=False)]
    qs = np.linspace(0, 1, bins + 1)[1:-1]
    return np.stack([np.nanquantile(s[:, j], qs) if np.isfinite(s[:, j]).any() else np.zeros(len(qs)) for j in range(F.shape[1])]).astype(np.float32)


def binize(F, edges):
    """分箱：缺失记 0，其余 1..bins（0 号箱只装缺失，分裂时自然学到缺失的去向）。"""
    B = np.zeros(F.shape, np.uint8)
    for j in range(F.shape[1]):
        x = F[:, j]
        ok = np.isfinite(x)
        B[ok, j] = (np.searchsorted(edges[j], x[ok], side="right") + 1).astype(np.uint8)
    return B


def rank_gauss_sec(y, dates):
    codes = pd.factorize(dates)[0]
    return T.rank_gauss(y, codes)


class GBDT:
    """逐层生长的直方图梯度提升（平方损失）。树 = 每层每个节点的 (特征, 阈值箱)，叶子值数组。"""

    def __init__(self, n_feat, nb, depth, lr, lam, min_leaf):
        self.nf, self.nb, self.depth, self.lr, self.lam, self.min_leaf = n_feat, nb, depth, lr, lam, min_leaf
        self.trees = []          # (feat (2^D−1,), thr (2^D−1,), leaf (2^D,))

    def _route(self, Xb, feat, thr):
        nid = torch.zeros(Xb.shape[0], dtype=torch.long, device=Xb.device)
        for lvl in range(self.depth):
            base = (1 << lvl) - 1
            f = feat[base + nid]
            t = thr[base + nid]
            x = Xb.gather(1, f.clamp(min=0).unsqueeze(1)).squeeze(1).long()
            go = (x > t) & (f >= 0)
            nid = 2 * nid + go.long()
        return nid

    def fit_tree(self, Xb, r, cols):
        """Xb (n, F) uint8 on device；r 残差 (n,)；cols 本棵树可用的特征下标 (Fs,)。"""
        n, D, nb = Xb.shape[0], self.depth, self.nb
        Fs = len(cols)
        nid = torch.zeros(n, dtype=torch.long, device=Xb.device)
        feat = torch.full((2 ** D - 1,), -1, dtype=torch.long, device=Xb.device)
        thr = torch.zeros(2 ** D - 1, dtype=torch.long, device=Xb.device)
        Xs = Xb[:, cols].long()                                            # (n, Fs)
        fidx = torch.arange(Fs, device=Xb.device).unsqueeze(0)
        for lvl in range(D):
            nn_ = 1 << lvl
            size = nn_ * Fs * nb
            G = torch.zeros(size, device=Xb.device); C = torch.zeros(size, device=Xb.device)
            step = max(1, 40_000_000 // Fs)
            for s0 in range(0, n, step):
                sl = slice(s0, s0 + step)
                idx = ((nid[sl].unsqueeze(1) * Fs + fidx) * nb + Xs[sl]).reshape(-1)
                G.index_add_(0, idx, r[sl].unsqueeze(1).expand(-1, Fs).reshape(-1))
                C.index_add_(0, idx, torch.ones(idx.shape[0], device=Xb.device))
            G = G.view(nn_, Fs, nb); C = C.view(nn_, Fs, nb)
            GL, CL = G.cumsum(2), C.cumsum(2)
            Gt, Ct = GL[:, :, -1:], CL[:, :, -1:]
            GR, CR = Gt - GL, Ct - CL
            gain = GL ** 2 / (CL + self.lam) + GR ** 2 / (CR + self.lam) - Gt ** 2 / (Ct + self.lam)
            ok = (CL >= self.min_leaf) & (CR >= self.min_leaf)
            gain = torch.where(ok, gain, torch.full_like(gain, -1.0))
            best = gain.view(nn_, -1).max(1)
            bf, bb = best.indices // nb, best.indices % nb
            good = best.values > 0
            base = nn_ - 1
            feat[base:base + nn_] = torch.where(good, cols[bf], torch.full_like(bf, -1))
            thr[base:base + nn_] = bb
            f = feat[base + nid]
            x = Xb.gather(1, f.clamp(min=0).unsqueeze(1)).squeeze(1).long()
            nid = 2 * nid + ((x > thr[base + nid]) & (f >= 0)).long()
        L = 2 ** D
        sG = torch.zeros(L, device=Xb.device).index_add_(0, nid, r)
        sC = torch.zeros(L, device=Xb.device).index_add_(0, nid, torch.ones_like(r))
        leaf = self.lr * sG / (sC + self.lam)
        self.trees.append((feat, thr, leaf))
        return leaf[nid]

    def predict_tree(self, Xb, k):
        feat, thr, leaf = self.trees[k]
        return leaf[self._route(Xb, feat, thr)]

    def to_json(self):
        return {"depth": self.depth, "lr": self.lr, "trees": [[t[0].tolist(), t[1].tolist(), [float(v) for v in t[2].tolist()]] for t in self.trees]}


def build(sd, ed, mode, stats=None, inst=None):
    X, y, keys, st = T.e04_build_dataset(fb, sd, ed, mode, inst, stats)
    return X, y, keys, st


def ic_by_sec(pred, y, dates):
    d = pd.DataFrame({"d": dates, "p": pred, "y": y}).dropna()
    return d.groupby("d")[["p", "y"]].apply(lambda g: g["p"].corr(g["y"], method="spearman")).mean()


torch.manual_seed(args.seed)
if args.small:
    tr_rng, va_rng, te_rng = ("2022-01-01 00:00:00", "2022-02-28 23:59:59"), ("2023-01-01 00:00:00", "2023-01-31 23:59:59"), None
else:
    tr_rng, va_rng, te_rng = ("2020-01-01 00:00:00", "2022-12-31 23:59:59"), ("2023-01-01 00:00:00", "2023-12-31 23:59:59"), ("2024-01-01 00:00:00", "2024-12-31 23:59:59")
inst_tr = fb.pool(*tr_rng)
inst_tr = inst_tr[:300] if args.small else inst_tr
t0 = time.time()
Xtr, ytr, ktr, stats = build(*tr_rng, "train", None, inst_tr)
_ST[0] = stats
Ftr = flatten(Xtr); del Xtr
if args.tgt == "e16":                                                            # 与 Transformer 训练完全同一目标（在含 09:30 的全部行上算，再去掉 09:30）
    T.TARGET, T.TARGET_AUX_W, T.TARGET_AUX_N, T.TRAIN_NEUTRAL = "cs_rank", args.aux_w, 2, "proxy_ind"
    yt = T.make_target(ytr, ktr)
    yt = (yt + T.TARGET_AUX_W * T.aux_target(ytr, ktr)).astype(np.float32)
    style = T.neutral_matrix(ktr, T.proxy_table(fb, tr_rng[0], tr_rng[1], inst_tr))
    yt = T.style_residual_np(yt, style, ktr["date"].values).astype(np.float32)
    del style
else:
    yt = None
keep = (ktr["date"].dt.strftime("%H%M") != "0930").to_numpy() & np.isfinite(ytr)
Ftr, ktr, ytr = Ftr[keep], ktr[keep].reset_index(drop=True), ytr[keep]
yt = yt[keep] if yt is not None else None
say("训练集", Ftr.shape, "构建 %.0fs" % (time.time() - t0), "设备", dev)
edges = make_edges(Ftr, args.bins - 1, args.seed)
Btr = torch.from_numpy(binize(Ftr, edges)).to(dev); del Ftr
tgt = torch.from_numpy(np.nan_to_num(yt) if yt is not None else rank_gauss_sec(ytr, ktr["date"].to_numpy())).float().to(dev)


def prep_eval(rng):
    inst = fb.pool(*rng)
    inst = inst[:300] if args.small else inst
    X, _, k, _ = build(*rng, "val", stats, inst)
    F = flatten(X); del X
    lab = fb.labels(*rng, inst)[["date", "instrument", "y"]]
    k = k.merge(lab, on=["date", "instrument"], how="left")
    return torch.from_numpy(binize(F, edges)).to(dev), k


Bva, kva = prep_eval(va_rng)
say("验证集", tuple(Bva.shape))
models = []
n = Btr.shape[0]
bag_va = torch.zeros(Bva.shape[0], device=dev)
for b in range(args.bags):
    rs = np.random.RandomState(args.seed + 1000 * b)
    m = GBDT(Btr.shape[1], args.bins, args.depth, args.lr, args.lam, args.min_leaf)
    pred_tr = torch.zeros(Btr.shape[0], device=dev)
    pred_va = torch.zeros(Bva.shape[0], device=dev)
    for k in range(args.trees):
        rows = torch.from_numpy(rs.choice(n, int(n * args.rows), replace=False)).to(dev)
        cols = torch.from_numpy(np.sort(rs.choice(Btr.shape[1], max(1, int(Btr.shape[1] * args.cols)), replace=False))).to(dev)
        r = tgt[rows] - pred_tr[rows]
        m.fit_tree(Btr[rows], r, cols)
        pred_tr += m.predict_tree(Btr, k)
        pred_va += m.predict_tree(Bva, k)
        if (k + 1) % 100 == 0 or k == 0:
            pv = pred_va.cpu().numpy()
            say("袋 %d 树 %d  训练 MSE %.4f  验证 IC %.4f  %.0fs" % (b, k + 1, float(((tgt - pred_tr) ** 2).mean()), ic_by_sec(pv, kva["y"].to_numpy(), kva["date"].to_numpy()), time.time() - t0))
    models.append(m)
    bag_va += pred_va
    say("袋 %d 完成；%d 袋平均的验证 IC %.4f" % (b, b + 1, ic_by_sec(bag_va.cpu().numpy(), kva["y"].to_numpy(), kva["date"].to_numpy())))
del pred_tr


def dump(B, k, fn):
    p = torch.zeros(B.shape[0], device=dev)
    for m in models:
        for j in range(len(m.trees)):
            p += m.predict_tree(B, j)
    out = k[["date", "instrument"] + T.CHAR_COLS].copy()
    out["factor"] = p.cpu().numpy() / len(models)
    out.to_parquet(os.path.join(OUT, fn), index=False)


dump(Bva, kva, "val_pred.parquet")
if te_rng:
    del Bva
    Bte, kte = prep_eval(te_rng)
    dump(Bte, kte, "pred_2024.parquet")
json.dump({"args": vars(args), "edges": edges.tolist(), "models": [m.to_json() for m in models]}, open(os.path.join(OUT, "gbdt_model.json"), "w"))
say("完成 %.0fs" % (time.time() - t0))
