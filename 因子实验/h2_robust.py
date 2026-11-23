# -*- coding: utf-8 -*-
"""第十五轮 J1／J2／J3：H2（浅层 GBDT 非线性堆叠）的稳健性、低换手版本、低换手头部族。

判定口径：一律 engine.Core.evaluate（c.evaluate），五个窗口逐项与**本地 1397**
（`v_D2OPT3_raw`／`v_D2OPT3O21_raw`／`v_D2OPT3O20_raw` 线性定权解）对比。

可扰动项：
  --seed       拟合随机种子（J1.1）
  --no-slot    去掉 7 个截面时刻 one-hot（J1.2）
  --fit        拟合窗 2020-2022 / 2021-2022（J1.3）
  --trees/--depth  树的规模（J1.4）
  --ewma       快变量轴先做 EWMA（J2.1，α 为“历史保留权重”，越大越平滑）
  --heads-only 只用他的头部族 + 1363（J3）
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
from h2_stack import GBDT, bin_edges, to_bins, mask_of, OUT, NBINS  # noqa: E402

SLOTS = (1000, 1030, 1100, 1130, 1330, 1400, 1430)
HEAD_NAMES = ("s161", "s155", "s139")
ROW5 = (("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2))
EXTRA = []
DROP = []
H1363_SUB = []                                                            # 10-18：H1363 只扣这些根（空 = 扣 DROP 里的全部成交笔数类根）；平台上 N_dir 活着，只扣 Y_sd5、X_n5d
PF1363 = [False]                                                           # Claude C2：H1363 输入减去被 drop 的成交笔数类根（平台上它们为零）
W1363 = {"L_lab10": 0.20, "M_q_lin": 0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": 0.10,
         "Y_sd5": 0.10, "X_n5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}


_K1AN = None


def extra_series(G, nm):
    """extra 列：优先用 vc_tmp/h1_<nm>.npy；没有就按 k1_anchor 的口径现算（当日不变锚）。"""
    global _K1AN
    p = OUT / ("h1_%s.npy" % nm)
    if nm.startswith(("MKT_", "MKD_")) and p.exists():                             # F11：截面状态特征，原始值（每截面一个值，排名会变成常数）
        return np.nan_to_num(np.load(p), nan=0.0)
    if p.exists():
        return np.nan_to_num(O.rc_of(np.load(p), G.c.date_id, G.c.n_dates), nan=0.0)
    import k1_anchor as K
    if _K1AN is None:
        _K1AN = K.make_anchors(K.load_frames())
    if nm not in _K1AN:
        raise SystemExit("extra 列 %s 既没有 npy 也不在 k1_anchor 的锚里" % nm)
    z930 = G.c.slot_of_date[G.c.date_id] == 930
    return np.nan_to_num(K.rank_of(G.c, _K1AN[nm], z930)[1], nan=0.0)


def ewma_col(x, inst, alpha):
    """α = 历史保留权重（与 root_eval.ewma_along 同义）。"""
    x = np.asarray(x, float)
    o = np.argsort(inst, kind="stable")
    e = pd.Series(x[o]).groupby(inst[o]).ewm(alpha=float(1.0 - alpha), adjust=False).mean().to_numpy()
    out = np.empty_like(e)
    out[o] = e
    return np.nan_to_num(out, nan=0.0)


def s152_axis(G, ix):
    c = G.c
    rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
    return np.nan_to_num(O.axis(c, ("fm_", "o21_", "o20_")[ix] + "s152", *rng), nan=0.0)


def anchor_cols(G, ix, names):
    """把 1398（G1B_B1）／1399（G4LOW）这两个"我们最好的低换手组合"的秩当额外输入列。"""
    from h1_stack import build as h1build, D1B, g4low_delta
    out = []
    for nm in names:
        if nm == "1398":
            B0 = h1build(G, ix, "s161", D1B)
        elif nm == "1399":
            B0 = h1build(G, ix, "1363", g4low_delta())
        else:
            raise SystemExit("未知锚 %s" % nm)
        B0[np.asarray(G.z930)] = 0.0
        out.append(np.nan_to_num(O.rc_of(B0, G.c.date_id, G.c.n_dates), nan=0.0))
    return out


EXTRA_ONLY = [False]                                                       # 10-19 D2：只用 --extra 特征与时段哑变量（盘口树，新方向）
NO_H1363 = [False]                                                         # 10-19 T2：特征里不放 1363 合成值


def build_X(G, ix, extra=(), slots=True, heads_only=False, ewma=0.0, anchors=()):
    c = G.c
    cols = []
    fast = []
    if EXTRA_ONLY[0]:
        if slots:
            slot = c.slot_of_date[c.date_id]
            for s in SLOTS:
                cols.append(np.where(slot == s, 1.0, 0.0))
        for nm in extra:
            cols.append(extra_series(G, nm))
        return np.column_stack([np.nan_to_num(np.asarray(x, np.float32), nan=0.0) for x in cols])
    if heads_only:
        for nm in HEAD_NAMES:
            cols.append(G.axis(nm, ix))
        fast = [True] * len(cols)
    else:
        for lab, _, _, _, _ in O.CAND:
            if lab in DROP:
                continue
            cols.append(G.axis(lab, ix))
        fast = [True] * len(cols)
    if NO_H1363[0]:                                                        # T2（10-19）：去掉 1363 合成值这一列（490 系公榜兑现差），只留各根与头部
        h1363 = None
    else:
        h1363 = np.nan_to_num(np.asarray(G.B[("1363", ix)], float), nan=0.0)
    if h1363 is not None:
        if PF1363[0]:                                                      # 平台一致：1363 内部的成交笔数类根在平台上为零
            h1363 = h1363 - sum(W1363[lab] * G.axis(lab, ix) for lab in (H1363_SUB or DROP) if lab in W1363)
        cols.append(h1363)
        fast.append(True)
    if not heads_only:
        cols.append(s152_axis(G, ix))
        fast.append(True)
    if slots and not heads_only:
        slot = c.slot_of_date[c.date_id]
        for s in SLOTS:
            cols.append(np.where(slot == s, 1.0, 0.0))
        fast.extend([False] * len(SLOTS))
    for nm in extra:
        cols.append(extra_series(G, nm))
        fast.append(False)
    if anchors:
        for x in anchor_cols(G, ix, anchors):
            cols.append(x)
            fast.append(False)
    if ewma:
        cols = [ewma_col(x, c.inst, ewma) if f else np.asarray(x, float) for x, f in zip(cols, fast)]
    return np.column_stack([np.nan_to_num(np.asarray(x, np.float32), nan=0.0) for x in cols])


class Ref1397:
    """本地 1397 线性定权解的五个窗口指标。"""

    def __init__(self):
        self.A = {"2023": R.load("v_D2OPT3_raw")[0], "2024": R.load("v_D2OPT3_raw")[0],
                  "2021": R.load("v_D2OPT3O21_raw")[0], "2022": R.load("v_D2OPT3O21_raw")[0],
                  "2020": R.load("v_D2OPT3O20_raw")[0]}
        self.cache = {}

    def get(self, c, wn):
        if wn not in self.cache:
            st, en = [q for q in WINS if q[0] == wn][0][1:3]
            self.cache[wn] = c.evaluate(np.asarray(self.A[wn], float), start=st, end=_en(en))[0]
        return self.cache[wn]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--trees", type=int, default=100)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--sub", type=float, default=0.08)
    ap.add_argument("--lr", type=float, default=0.06)
    ap.add_argument("--no-slot", action="store_true")
    ap.add_argument("--heads-only", action="store_true")
    ap.add_argument("--fit", default="2020-2022")
    ap.add_argument("--ewma", type=float, default=0.0)
    ap.add_argument("--boost", default="", help="在 1398（G1B_B1）的秩底上做残差提升；填 1398 表示开启")
    ap.add_argument("--boost-lam", type=float, default=1.0)
    ap.add_argument("--boost-mode", default="resid", choices=["resid", "init"],
                    help="resid=把 1398 秩当目标减掉（旧口径）；init=把 1398 秩当 GBDT 的初始值（正确的加性提升）")
    ap.add_argument("--extra", default="")
    ap.add_argument("--drop", default="", help="从 15 个分量轴里剔除的列（如 Y_sd5,X_n5d）")
    ap.add_argument("--anchor", default="", help="额外输入列：1398,1399（各自组合的截面秩）")
    ap.add_argument("--tag", default="rob")
    ap.add_argument("--save", action="store_true", help="按年导出 npy（h2_<tag>_<year>.npy）")
    ap.add_argument("--save-json", default="")
    ap.add_argument("--quiet-trees", action="store_true")
    ap.add_argument("--keep-float", action="store_true")
    ap.add_argument("--h1363-pf", action="store_true", help="H1363 输入减去被 drop 的成交笔数类根（平台一致）")
    ap.add_argument("--aux-w", type=float, default=0.0, help="F2：多期目标权重——目标 = 秩(当前标签) + w·秩(同日之后 --aux-n 个截面标签之和)，只改训练目标")
    ap.add_argument("--aux-n", type=int, default=2)
    ap.add_argument("--no-h1363", action="store_true", help="T2（10-19）：特征里不放 1363 合成值")
    ap.add_argument("--extra-only", action="store_true", help="10-19 D2：只用 --extra 特征与时段哑变量")
    ap.add_argument("--aux-rest", action="store_true", help="F9：多期目标改为同日之后全部截面标签之和（缺的截面记 0，当日最后一个截面没有）")
    ap.add_argument("--target-gauss", action="store_true", help="10-18：目标的截面秩先转标准正态分位数（端到端 R2a 的做法），多期目标同样处理")
    ap.add_argument("--target-resid", action="store_true", help="10-20：训练目标逐截面对风格 + 行业取残差（与评测机 core.residual 同一实现），树只学评分时剩下的部分")
    ap.add_argument("--mono-free", default="", help="与 --mono 合用：这些特征不约束（名字：各根标签、H1363、s152、额外特征名）")
    ap.add_argument("--mono", action="store_true", help="10-20：单调约束，方向 = 拟合区间内各特征与目标相关的符号（时段哑变量不约束）")
    ap.add_argument("--h1363-sub", default="", help="与 --h1363-pf 合用：H1363 只扣这些根（如 Y_sd5,X_n5d），其余 drop 的根只从特征里去掉")
    a = ap.parse_args()

    EXTRA.extend([x for x in a.extra.split(",") if x])
    EXTRA_ONLY[0] = a.extra_only
    NO_H1363[0] = a.no_h1363
    DROP.extend([x for x in a.drop.split(",") if x])
    PF1363[0] = bool(a.h1363_pf)
    H1363_SUB[:] = [x for x in a.h1363_sub.split(",") if x]
    fspan = {"2020-2022": ("2020-01-01", "2022-12-31"),
             "2021-2022": ("2021-01-01", "2022-12-31"),
             "2020-2023": ("2020-01-01", "2023-12-31"),
             "2020-2024": ("2020-01-01", "2024-12-31")}[a.fit]                 # T1（10-19）：交付用全量拟合（公榜区间不与本地数据重叠）                 # F15（10-19）：拟合到 2023，只看 2024（往后看的样本外）

    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    t0 = time.time()
    print("配置 seed=%d trees=%d depth=%d sub=%.2f lr=%.2f slot=%s heads_only=%s fit=%s..%s ewma=%.2f extra=%s drop=%s"
          % (a.seed, a.trees, a.depth, a.sub, a.lr, not a.no_slot, a.heads_only, fspan[0], fspan[1], a.ewma, EXTRA, DROP),
          flush=True)
    ANCH = [x for x in a.anchor.split(",") if x]
    Xs = {ix: build_X(G, ix, extra=EXTRA, slots=not a.no_slot, heads_only=a.heads_only, ewma=a.ewma,
                      anchors=ANCH)
          for ix in (0, 1, 2)}
    print("特征 %d 列（锚 %s），构造 %.0fs" % (Xs[0].shape[1], ANCH, time.time() - t0), flush=True)

    kd = pd.to_datetime(c.key_date)
    yr = kd.year.to_numpy()
    Xf = np.where((yr >= 2021)[:, None], Xs[1], Xs[2])
    m_fit = mask_of(c, fspan)
    if a.target_resid:
        m_fit &= np.isfinite(c.residual(np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)))
    y = np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)
    if a.target_gauss:                                                     # 秩（−0.5..0.5）→ 标准正态分位数
        from scipy.special import erfinv
        y = np.sqrt(2.0) * erfinv(np.clip(2.0 * y, -0.999998, 0.999998))
    if a.aux_w > 0:                                                        # F2：多期目标（端到端辅助目标的因子版），特征不变
        dfa = pd.DataFrame({"row": np.arange(c.n_rows), "inst": c.inst, "sec": c.date_id,
                            "day": c.day_of_date[c.date_id], "y": c.y}).sort_values(["inst", "sec"])
        ga = dfa.groupby(["inst", "day"], sort=False)["y"]
        if a.aux_rest:
            sh = [ga.shift(-k) for k in range(1, 9)]
            nxt = sum(x.fillna(0.0) for x in sh).where(sh[0].notna())
        else:
            nxt = sum(ga.shift(-k) for k in range(1, a.aux_n + 1))
        ysum = np.full(c.n_rows, np.nan)
        ysum[dfa["row"].to_numpy()] = nxt.to_numpy()
        ya = np.nan_to_num(O.rc_of(ysum, c.date_id, c.n_dates), nan=0.0)
        if a.target_gauss:
            from scipy.special import erfinv
            ya = np.sqrt(2.0) * erfinv(np.clip(2.0 * ya, -0.999998, 0.999998))
        y = y + a.aux_w * ya
        print("多期目标：w=%.2f n=%s，之后截面标签和的非空率 %.3f" % (a.aux_w, "rest" if a.aux_rest else a.aux_n, np.isfinite(ysum).mean()), flush=True)
    if a.target_resid:                                                     # 10-20：评测先剔除风格与行业，目标同口径
        yres = c.residual(y)
        ok_r = np.isfinite(yres)
        print("目标风格剔除：有效行 %.3f，与原目标的相关 %.3f" % (ok_r.mean(), np.corrcoef(y[ok_r], yres[ok_r])[0, 1]), flush=True)
        y = np.nan_to_num(yres * 0.2887, nan=0.0)
    R0 = {}
    BASE_IX = {}
    if a.boost:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from h1_stack import build as h1build, D1B
        for ix in (0, 1, 2):
            B0 = h1build(G, ix, "s161", D1B)
            B0[z930] = 0.0
            BASE_IX[ix] = np.nan_to_num(B0, nan=0.0)
            R0[ix] = np.nan_to_num(O.rc_of(B0, c.date_id, c.n_dates), nan=0.0)
            mm, _ = c.evaluate(np.where(z930, 0.0, B0), start={"0": "2023-01-01", "1": "2021-01-01", "2": "2020-01-01"}[str(ix)],
                               end=_en({"0": "2024-12-31", "1": "2022-12-31", "2": "2020-12-31"}[str(ix)]))
            print("  底座 1398 轴 ix=%d IC %.5f IR %.4f 压力 %.4f 换手 %.4f"
                  % (ix, mm["ic_mean"], mm["ic_ir"], mm["stress"], mm["turnover"]), flush=True)
        Rf = np.where((yr >= 2021), R0[1], R0[2])
        if a.boost_mode == "resid":
            y = y - Rf
        else:
            print("提升模式 init：F0 = 1398 秩，树拟合 标签−底座 的残差", flush=True)
        print("提升模式：目标 = 标签秩 − 1398 秩（底座 ic %.5f）" % 0.0, flush=True)
    else:
        Rf = None
    print("拟合行 %d（窗 %s）" % (m_fit.sum(), a.fit), flush=True)
    t0 = time.time()
    E = bin_edges(Xf[m_fit])
    mono = None
    if a.mono:                                                             # 10-20：单调约束方向只用拟合区间
        fi = np.flatnonzero(m_fit)
        fi = fi[np.random.default_rng(0).choice(len(fi), min(len(fi), 2000000), replace=False)]
        mono = np.zeros(Xf.shape[1], np.int64)
        for j in range(Xf.shape[1]):
            xj = Xf[fi, j].astype(float)
            if np.isin(xj, (0.0, 1.0)).all():
                continue
            mono[j] = 1 if np.corrcoef(xj, y[fi])[0, 1] > 0 else -1
        free = [x for x in a.mono_free.split(",") if x]
        if free:                                                           # 分组消融：指定的特征不约束
            names = [lab for lab, _, _, _, _ in O.CAND if lab not in DROP] + ([] if NO_H1363[0] else ["H1363"]) + ["s152"] \
                + (["slot%d" % s0 for s0 in SLOTS] if not a.no_slot else []) + list(EXTRA)
            assert len(names) == Xf.shape[1] and not a.heads_only and not ANCH, (len(names), Xf.shape[1])
            assert all(x in names for x in free), free
            mono[[names.index(x) for x in free]] = 0
            print("不约束", free, flush=True)
        print("单调约束方向", mono.tolist(), flush=True)
    Bf = to_bins(Xf, E)
    Bs = {}
    if not getattr(a, "keep_float", False):
        for ix in (0, 1, 2):
            Bs[ix] = to_bins(Xs[ix], E)
        del Xf, Xs
        import gc
        gc.collect()
        print("已转分箱并释放浮点特征（低内存模式）", flush=True)
    gf = GBDT(n_trees=a.trees, depth=a.depth, sub=a.sub, lr=a.lr, seed=a.seed, mono=mono)
    if a.boost and a.boost_mode == "init":
        gf.fit(Bf[m_fit], y[m_fit], verbose=0 if a.quiet_trees else 25, F0=Rf[m_fit])
    else:
        gf.fit(Bf[m_fit], y[m_fit], verbose=0 if a.quiet_trees else 25)
    print("拟合完成 %.1fs" % (time.time() - t0), flush=True)

    ref = Ref1397()
    print("%-10s|%s" % ("窗口", "".join(" %-28s" % w for w, _ in ROW5)), flush=True)
    print("%-10s|%s" % ("", "".join(" %-28s" % "IC/IR/多空/压力/换手" for _ in ROW5)), flush=True)
    r0 = ref.get(c, "2023")
    print("%-10s|%s" % ("1397", "".join(
        " %-28s" % ("%.5f/%.4f/%.1f/%.4f/%.4f" % tuple(ref.get(c, w)[k] for k in
                                                     ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"))) for w, _ in ROW5)),
        flush=True)
    summ = {}
    for wn, ix in ROW5:
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m = mask_of(c, (st, en))
        B = Bs[ix] if Bs else to_bins(Xs[ix], E)
        pr = np.full(c.n_rows, np.nan)
        pr[m] = gf.predict(B[m]) + (a.boost_lam * R0[ix][m] if a.boost else 0.0)
        pr = np.where(z930, 0.0, pr)
        mm, _ = c.evaluate(pr, start=st, end=_en(en))
        rr = ref.get(c, wn)
        summ[wn] = {k: float(mm[k]) for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}
        if a.boost:
            bb, _ = c.evaluate(np.asarray(BASE_IX[ix], float), start=st, end=_en(en))
            summ[wn]["base1398"] = {k: float(bb[k]) for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}
            print("%-10s| 相对 1398: ΔIC %+.5f ΔIR %+.4f Δ多空 %+.1f Δ压力 %+.4f Δ换手 %+.4f"
                  % ("  vs1398", mm["ic_mean"] - bb["ic_mean"], mm["ic_ir"] - bb["ic_ir"],
                     mm["ls_sharpe"] - bb["ls_sharpe"], mm["stress"] - bb["stress"],
                     mm["turnover"] - bb["turnover"]), flush=True)
        print("%-10s| %s   ΔIC %+.5f ΔIR %+.4f Δ多空 %+.1f Δ压力 %+.4f Δ换手 %+.4f" % (
            a.tag + "_" + wn, "%.5f/%.4f/%.1f/%.4f/%.4f" % tuple(mm[k] for k in
                                                             ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")),
            mm["ic_mean"] - rr["ic_mean"], mm["ic_ir"] - rr["ic_ir"], mm["ls_sharpe"] - rr["ls_sharpe"],
            mm["stress"] - rr["stress"], mm["turnover"] - rr["turnover"]), flush=True)
        if a.save:
            np.save(OUT / ("h2_%s_%s.npy" % (a.tag, wn)), np.nan_to_num(pr, nan=0.0))
    if a.save_json:
        json.dump({"model": gf.export(), "edges": np.asarray(E).tolist(), "fit_span": list(fspan),
                   "extra": list(EXTRA), "no_slot": a.no_slot, "heads_only": a.heads_only, "ewma": a.ewma,
                   "anchor": list(ANCH), "feat_order": (list(HEAD_NAMES) + ([] if a.heads_only else ["s152"])) if a.heads_only else
                                 ([l for l, *_ in O.CAND if l not in DROP] + ["H1363", "H152"]
                                  + ([] if a.no_slot else ["slot%s" % s for s in SLOTS]) + list(EXTRA)
                                  + ["anchor%s" % x for x in ANCH])},
                  open(a.save_json, "w"), default=float)
        print("模型已导出 %s" % a.save_json, flush=True)
    print("ROB %s %s" % (a.tag, json.dumps(summ)), flush=True)


if __name__ == "__main__":
    main()
