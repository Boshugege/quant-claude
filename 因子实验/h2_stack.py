# -*- coding: utf-8 -*-
"""第十四轮 H2：非线性堆叠（纯 numpy 浅层 GBDT）＋线性交互对照。

输入：D2 的 15 个分量轴 + 我们头部 1363 + 他的头部 s152 + 截面时刻（7 个 one-hot）。
目标：官方标签的截面（date_id）秩。
只在 2020-01~2022-12 拟合（2020 用 o20_ 轴、2021-22 用 o21_ 轴），2023-01~2024-12 验证（fm_ 轴）。
环境无 sklearn/lightgbm → 纯 numpy 直方图 GBDT（深度≤3、树≤200、lr 小、L2 正则、行/列抽样），
模型可导出 JSON（每棵树的 split(feature, bin) 与叶子值），供 Claude 做平台代码版。
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402
from g_go import GO, WINS, _en  # noqa: E402
from h1_stack import build  # noqa: E402


def en_ts(s):
    return _en(s)

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
FITSPAN = ("2020-01-01", "2022-12-31")
NBINS = 32


EXTRA = []


def build_X(G, ix):
    c = G.c
    cols = [G.axis(lab, ix) for lab, _, _, _, _ in O.CAND]
    cols.append(np.asarray(G.B[("1363", ix)], float))
    rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
    cols.append(np.nan_to_num(O.axis(c, ("fm_", "o21_", "o20_")[ix] + "s152", *rng), nan=0.0))
    slot = c.slot_of_date[c.date_id]
    for s in (1000, 1030, 1100, 1130, 1330, 1400, 1430):
        cols.append(np.where(slot == s, 1.0, 0.0))
    for nm in EXTRA:
        x = np.load(OUT / ("h1_%s.npy" % nm))
        cols.append(np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0))
    return np.column_stack([np.nan_to_num(np.asarray(x, np.float32), nan=0.0) for x in cols])


def mask_of(c, span):
    kd = pd.to_datetime(c.key_date)
    e = pd.Timestamp(span[1]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    m = (kd >= pd.Timestamp(span[0])) & (kd <= e) & np.isfinite(c.y)
    return m & (c.slot_of_date[c.date_id] != 930)


def bin_edges(X):
    E = np.empty((X.shape[1], NBINS - 1), np.float32)
    for j in range(X.shape[1]):
        E[j] = np.quantile(X[:, j], np.linspace(0, 1, NBINS + 1)[1:-1])
    return E


def to_bins(X, E):
    B = np.empty(X.shape, np.uint8)
    for j in range(X.shape[1]):
        B[:, j] = np.searchsorted(E[j], X[:, j], side="left").astype(np.uint8)
    return B


def _tree_tables(tree):
    maxid = max(x[1] for x in tree)
    LV = np.zeros(maxid + 1)
    SF = np.zeros(maxid + 1, np.int64)
    SK = np.zeros(maxid + 1, np.int64)
    SD = np.zeros(maxid + 1, np.int64)
    HS = np.zeros(maxid + 1, bool)
    for it in tree:
        if it[0] == "split":
            _, nid, f, k, d = it
            SF[nid], SK[nid], SD[nid], HS[nid] = f, k, d, True
        else:
            LV[it[1]] = it[2]
    return LV, SF, SK, SD, HS


class GBDT:
    def __init__(self, n_trees=120, depth=3, lr=0.08, min_leaf=3000, sub=0.12,
                 colsub=0.8, reg=200.0, seed=7, mono=None):
        self.p = dict(n_trees=n_trees, depth=depth, lr=lr, min_leaf=min_leaf, sub=sub,
                      colsub=colsub, nbins=NBINS, reg=reg, seed=seed)
        self.mono = None if mono is None else np.asarray(mono, np.int64)    # 10-20 单调约束：+1 递增、−1 递减、0 不约束（LightGBM basic：上下界逐层下传）

    def fit(self, B, y, verbose=10, F0=None):
        rng = np.random.default_rng(self.p["seed"])
        n = len(y)
        self.trees = []
        if F0 is None:
            self.base = float(y.mean())
            F = np.full(n, self.base)
        else:
            self.base = 0.0
            F = np.asarray(F0, float).copy()
        ns = max(int(n * self.p["sub"]), 20000)
        t0 = time.time()
        for t in range(self.p["n_trees"]):
            rows = rng.choice(n, ns, replace=False)
            r = y[rows] - F[rows]
            tree = self._grow(B[rows], r, rng)
            self.trees.append(tree)
            self._apply(B, tree, F, self.p["lr"])
            if verbose and (t + 1) % verbose == 0:
                print("   tree %d/%d rss %.5f  %.1fs" % (t + 1, self.p["n_trees"],
                                                         float((r ** 2).mean()), time.time() - t0), flush=True)
        return self

    def _grow(self, B, r, rng):
        nf = B.shape[1]
        p = self.p
        nodes = {0: (np.arange(len(r)), -np.inf, np.inf)}
        tree = []
        mono = self.mono
        leafv = lambda rows, lo, hi: float(np.clip(r[rows].sum() / (len(rows) + p["reg"]), lo, hi))  # noqa: E731
        for d in range(p["depth"]):
            new = {}
            for nid, (rows, lo, hi) in nodes.items():
                if len(rows) < 2 * p["min_leaf"]:
                    tree.append(("leaf", nid, leafv(rows, lo, hi)))
                    continue
                feats = rng.choice(nf, max(1, int(nf * p["colsub"])), replace=False)
                tot = len(rows)
                best = None
                for f in feats:
                    b = B[rows, f].astype(np.int64)
                    cnt = np.bincount(b, minlength=p["nbins"]).astype(np.float64)
                    syb = np.bincount(b, weights=r[rows], minlength=p["nbins"])
                    cl = np.cumsum(cnt)[:-1]
                    sl = np.cumsum(syb)[:-1]
                    cr = tot - cl
                    sr = syb.sum() - sl
                    valid = (cl >= p["min_leaf"]) & (cr >= p["min_leaf"])
                    if mono is not None and mono[f] != 0:                  # 单调：右（特征大）− 左 的符号须与约束一致
                        valid &= mono[f] * (sr / (cr + p["reg"]) - sl / (cl + p["reg"])) >= 0
                    if not valid.any():
                        continue
                    sc = np.where(valid, sl * sl / np.maximum(cl, 1) + sr * sr / np.maximum(cr, 1), -np.inf)
                    k = int(np.argmax(sc))
                    if best is None or sc[k] > best[0]:
                        best = (sc[k], f, k)
                if best is None:
                    tree.append(("leaf", nid, leafv(rows, lo, hi)))
                    continue
                _, f, k = best
                m = B[rows, f] <= k
                tree.append(("split", nid, f, int(k), d))
                bl, br = (lo, hi), (lo, hi)
                if mono is not None and mono[f] != 0:
                    wl = np.clip(r[rows[m]].sum() / (m.sum() + p["reg"]), lo, hi)
                    wr = np.clip(r[rows[~m]].sum() / ((~m).sum() + p["reg"]), lo, hi)
                    mid = 0.5 * (wl + wr)
                    bl, br = ((lo, min(hi, mid)), (max(lo, mid), hi)) if mono[f] > 0 else ((max(lo, mid), hi), (lo, min(hi, mid)))
                new[2 * nid + 1] = (rows[m],) + bl
                new[2 * nid + 2] = (rows[~m],) + br
            nodes = new
            if not nodes:
                break
        for nid, (rows, lo, hi) in nodes.items():
            tree.append(("leaf", nid, leafv(rows, lo, hi)))
        return tree

    def _apply(self, B, tree, F, lr):
        LV, SF, SK, SD, HS = _tree_tables(tree)
        node = np.zeros(len(B), np.int64)
        depth = int(SD[HS].max()) if HS.any() else -1
        for d in range(depth + 1):
            cur = HS[node]
            cur &= (SD[node] == d)
            if not cur.any():
                break
            ii = np.flatnonzero(cur)
            cmp = B[ii, SF[node[ii]]] <= SK[node[ii]]
            node[ii] = np.where(cmp, 2 * node[ii] + 1, 2 * node[ii] + 2)
        F += lr * LV[node]

    def predict(self, B):
        out = np.zeros(len(B))
        for tree in self.trees:
            self._apply(B, tree, out, 1.0)
        return out + self.base

    def export(self):
        return {"params": self.p, "base": self.base,
                "trees": [[list(x) for x in tr] for tr in self.trees]}


def ridge_fit(X, y, g, nd, lam=1.0):
    X = X.astype(np.float64)
    cnt = np.maximum(np.bincount(g, minlength=nd), 1).astype(float)
    for j in range(X.shape[1]):
        X[:, j] -= np.bincount(g, X[:, j], nd)[g] / cnt[g]
    ry = y - np.bincount(g, y, nd)[g] / cnt[g]
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ ry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--sub", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--save", default="")
    ap.add_argument("--no-ridge", action="store_true")
    ap.add_argument("--extra", default="")
    ap.add_argument("--tag", default="gbdt")
    a = ap.parse_args()
    G = GO()
    c = G.c
    EXTRA.extend([x for x in a.extra.split(",") if x])
    print("附加特征 %s" % EXTRA, flush=True)
    z930 = c.slot_of_date[c.date_id] == 930
    print("特征轴构造中…", flush=True)
    Xs = {ix: build_X(G, ix) for ix in (0, 1, 2)}
    kd = pd.to_datetime(c.key_date)
    yr = kd.year
    print("特征 %d 列" % Xs[0].shape[1], flush=True)
    Xf = np.where((yr >= 2021)[:, None], Xs[1], Xs[2])
    m_fit = mask_of(c, FITSPAN)
    print("拟合行 %d" % m_fit.sum(), flush=True)
    E = bin_edges(Xf[m_fit])
    Bf = to_bins(Xf, E)
    y = np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)

    rows = []
    for wn, ix in (("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2)):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m = mask_of(c, (st, en))
        rows.append((wn, ix, st, en, m))
        ref = {nm: G.ev(np.asarray(G.B[(nm, ix)], float), wn) for nm in ("1363", "s161")}
        print("%-6s 参考 1363 IC %.5f IR %.4f 压力 %.4f 换手 %.4f | s161 IC %.5f IR %.4f 换手 %.4f"
              % (wn, ref["1363"]["ic_mean"], ref["1363"]["ic_ir"], ref["1363"]["stress"], ref["1363"]["turnover"],
                 ref["s161"]["ic_mean"], ref["s161"]["ic_ir"], ref["s161"]["turnover"]), flush=True)

    gf = GBDT(n_trees=a.trees, depth=a.depth, sub=a.sub, lr=a.lr)
    print("拟合 GBDT …", flush=True)
    gf.fit(Bf[m_fit], y[m_fit])
    for wn, ix, st, en, m in rows:
        B = to_bins(Xs[ix], E)
        pr = np.full(c.n_rows, np.nan)
        pr[m] = gf.predict(B[m])
        pr = np.where(z930, 0.0, pr)
        mm, _ = c.evaluate(pr, start=st, end=en_ts(en))
        print("GBDT_%-6s IC %.5f IR %.4f LS %.1f 压力 %.4f 换手 %.4f" %
              (wn, mm["ic_mean"], mm["ic_ir"], mm["ls_sharpe"], mm["stress"], mm["turnover"]), flush=True)
        np.save(OUT / ("h2_%s_%s.npy" % (a.tag, wn)), np.nan_to_num(pr, nan=0.0))
    if a.save:
        json.dump({"model": gf.export(), "edges": np.asarray(E).tolist(), "fit_span": FITSPAN,
                   "extra": list(EXTRA), "feat_order": [l for l, *_ in O.CAND] + ["H1363", "H152"]
                   + ["slot%s" % s for s in (1000, 1030, 1100, 1130, 1330, 1400, 1430)] + list(EXTRA)},
                  open(a.save, "w"), default=float)
        print("模型已导出 %s" % a.save)

    if not a.no_ridge:
        print("\n线性交互对照（岭回归）…", flush=True)
        inter = [(15, 0), (15, 7), (16, 0), (16, 7)]  # 特征列布局不变时成立      # H1363/H152 × L_lab10/Z_mfr60
        gi = lambda X: np.column_stack([X] + [X[:, i] * X[:, j] for i, j in inter])
        Xi = gi(Xf)
        idx = np.flatnonzero(m_fit)
        beta = ridge_fit(Xi[idx], y[idx], c.date_id[idx], c.n_dates)
        for wn, ix, st, en, m in rows:
            sc = np.full(c.n_rows, np.nan)
            Xv = gi(Xs[ix])
            sc[m] = Xv[m] @ beta
            sc = np.where(z930, 0.0, sc)
            mm, _ = c.evaluate(sc, start=st, end=en_ts(en))
            print("RIDGE_%-6s IC %.5f IR %.4f LS %.1f 压力 %.4f 换手 %.4f" %
                  (wn, mm["ic_mean"], mm["ic_ir"], mm["ls_sharpe"], mm["stress"], mm["turnover"]), flush=True)
            np.save(OUT / ("h2_ridge_%s.npy" % wn), np.nan_to_num(sc, nan=0.0))
        print("岭回归系数：", np.round(beta, 4).tolist())


if __name__ == "__main__":
    main()
