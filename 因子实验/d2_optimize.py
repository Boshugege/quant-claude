# -*- coding: utf-8 -*-
"""D2 全局权重优化：把 1363 的九根 + 第二头部候选 + 小增量根放在一起重新定权。

B = 1363原始值 + Σ_j δ_j·rank(c_j)。δ=0 就是现在的 1363 贪心解。
- 拟合窗口 2020-01～2022-12（V1363O20/V1363O21 两个样本外 1363 值任务拼接，根用 o20_/o21_）；
- 验证窗口 2023-01～2024-12（主区间）。
方向由逐截面岭回归给出（标签居中分位排名 ~ 各候选居中分位排名，缺失填 0），再在 1 维尺度与
前 6 个坐标上做局部搜索。目标 J = IC_IR + 压力 − λ·换手（λ 默认 0.2）。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import engine as E  # noqa: E402

CAND = [
    ("L_lab10", "fm_1232", "o21_1232", "o20_1232", 0.75),
    ("M_q_lin", "fm_1236", "o21_1236", "o20_1236", 0.0),
    ("N_dir", "fm_1237", "o21_1237", "o20_1237", 0.0),
    ("W_vdir", "fm_1260", "o21_1260", "o20_1260", 0.0),
    ("W_zero", "fm_1263", "o21_1263", "o20_1263", 0.0),
    ("Y_sd5", "fm_1323", "o21_1323", "o20_1323", 0.0),
    ("X_n5d", "fm_1273", "o21_1273", "o20_1273", 0.0),
    ("Z_mfr60", "fm_1334", "o21_1334", "o20_1334", 0.0),
    ("Z_mfr90", "fm_1342", "o21_1342", "o20_1342", 0.0),
    ("s155", "fm_s155", "o21_s155", "o20_s155", 0.0),
    ("s139", "fm_s139", "o21_s139", "o20_s139", 0.0),
    ("s161", "fm_s161", "o21_s161", "o20_s161", 0.0),
    ("k1381", "fm_1381", "o21_1381", "o20_1381", 0.0),
    ("k1385", "fm_1385", "o21_1385", "o20_1385", 0.0),
    ("k1406", "fm_1406", "o21_1406", "o20_1406", 0.0),
]
W1363 = {"L_lab10": 0.20, "M_q_lin": 0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": 0.10,
         "Y_sd5": 0.10, "X_n5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}
FIT = ("2020-01-01", "2022-12-31")
VAL = ("2023-01-01", "2024-12-31")


def rc_of(vals, gid, nd):
    m = np.isfinite(vals)
    out = np.full(len(vals), np.nan)
    if m.any():
        rp, _ = E._group_rank_pct(gid[m], vals[m], nd)
        out[m] = rp - 0.5
    return out


def axis(c, name, start, end, alpha=0.0):
    """候选在该区间的居中分位排名轴；alpha>0 时先在 factor_keys 上做同一股票连续截面 EWMA。"""
    d = R.jobdir(name)
    kf = pd.read_parquet(d / "factor_keys.parquet")
    kd = kf["key_date"].to_numpy("datetime64[ns]").astype(np.int64)
    inst = kf["instrument_id"].to_numpy()
    fac = kf["factor"].to_numpy(float)
    if alpha and alpha > 0:
        o = np.lexsort((kd, inst))
        inst_s, kd_s, fac_s = inst[o], kd[o], fac[o]
        sm = np.empty(len(fac_s))
        cur, prev = np.nan, None
        for k in range(len(fac_s)):
            if inst_s[k] != prev:
                cur, prev = np.nan, inst_s[k]
            v = fac_s[k]
            if np.isfinite(v):
                cur = v if not np.isfinite(cur) else (alpha * cur + (1 - alpha) * v)
            sm[k] = cur
        f, _hit = c.align(kd_s, inst_s, sm)
    else:
        f, _hit = c.align(kd, inst, fac)
    _ok, did, _slot = R.base_rows(start, end)
    out = rc_of(np.asarray(f, float), did, c.n_dates)
    # 与 values 合成脚本一致：09:30 截面的 rank 轴置 0（否则合成值在该截面出现伪离散，
    # 评测机会多算一个 09:30 截面，实测把 2023 IC 拖低 0.008）
    out[c.slot_of_date[c.date_id] == 930] = 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--max-eval", type=int, default=70)
    a = ap.parse_args()
    c = R.core()
    nd = c.n_dates
    H20 = R.load("v_V1363O20_raw_wm0100")[0]
    H21 = R.load("v_V1363O21_raw_wm0100")[0]
    fitH = np.where(np.isfinite(H21), H21, H20)
    valH = R.load("v_S1284z1342_raw_wm0100")[0]
    print("fit H 有效 %d ; val H 有效 %d" % (np.isfinite(fitH).sum(), np.isfinite(valH).sum()))

    fitR, valR = {}, {}
    for lab, nm, nm21, nm20, al in CAND:
        r20 = axis(c, nm20, "2020-01-01", "2020-12-31", al)
        r21 = axis(c, nm21, "2021-01-01", "2022-12-31", al)
        fitR[lab] = np.where(np.isfinite(r21), r21, r20)
        valR[lab] = axis(c, nm, VAL[0], VAL[1], al)
    print("轴构造完成")

    def evaluate(H, Rs, w, span):
        B = np.asarray(H, float).copy()
        for lab, val in w.items():
            if val:
                B = B + val * np.nan_to_num(Rs[lab], nan=0.0)
        m, _s = c.evaluate(B, start=span[0], end=span[1])
        return m

    def J(m):
        return m["ic_ir"] + m["stress"] - a.lam * m["turnover"]

    m0 = evaluate(fitH, fitR, {}, FIT)
    v0 = evaluate(valH, valR, {}, VAL)
    greedy = {"s155": 0.25, "s139": 0.10, "k1381": 0.02, "k1385": 0.02, "k1406": 0.05}
    g1 = evaluate(fitH, fitR, greedy, FIT)
    g2 = evaluate(valH, valR, greedy, VAL)
    print("fit  δ=0    IC %.5f IR %.4f 压力 %.4f 换手 %.4f  J %.4f" % (m0["ic_mean"], m0["ic_ir"], m0["stress"], m0["turnover"], J(m0)))
    print("val  δ=0    IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (v0["ic_mean"], v0["ic_ir"], v0["stress"], v0["turnover"]))
    print("fit  贪心S4L IC %.5f IR %.4f 压力 %.4f 换手 %.4f  J %.4f" % (g1["ic_mean"], g1["ic_ir"], g1["stress"], g1["turnover"], J(g1)))
    print("val  贪心S4L IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (g2["ic_mean"], g2["ic_ir"], g2["stress"], g2["turnover"]))

    # ---- 方向：逐截面岭回归
    _ok, did, slot = R.base_rows(FIT[0], FIT[1])
    keep = _ok                      # 已含：标签有限 & 在拟合窗内 & 非 09:30
    idx = np.flatnonzero(keep)
    g = did[idx]
    labels = list(fitR)
    X = np.column_stack([np.nan_to_num(np.asarray(fitR[lab][idx], float)) for lab in labels])
    ry = rc_of(c.y[idx], g, nd)
    good = np.isfinite(ry)
    idx, g, X, ry = idx[good], g[good], X[good], ry[good]
    cnt = np.bincount(g, minlength=nd).astype(float)
    cnt = np.where(cnt > 0, cnt, 1.0)
    for j in range(X.shape[1]):
        X[:, j] -= np.bincount(g, X[:, j], nd)[g] / cnt[g]
    ry = ry - np.bincount(g, ry, nd)[g] / cnt[g]
    w_hat = np.linalg.solve(X.T @ X + 1.0 * np.eye(X.shape[1]), X.T @ ry)
    print("\n岭回归方向（回归样本 %d 行）" % len(idx))
    for lab, wv in sorted(zip(labels, w_hat), key=lambda t: -abs(t[1])):
        print("   %-8s %+.4f   （1363 现在 %+.3f）" % (lab, wv, W1363.get(lab, 0.0)))

    best = ({lab: 0.0 for lab in labels}, J(m0), m0)
    print("\n尺度搜索 δ = γ·w_hat")
    nei = 0
    for gam in (0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0):
        w = {lab: gam * wv for lab, wv in zip(labels, w_hat)}
        m = evaluate(fitH, fitR, w, FIT)
        nei += 1
        print("   γ=%.2f  IC %.5f IR %.4f 压力 %.4f 换手 %.4f → J %.4f" % (gam, m["ic_mean"], m["ic_ir"], m["stress"], m["turnover"], J(m)))
        if J(m) > best[1]:
            best = (w, J(m), m)
    order = [lab for lab, _ in sorted(zip(labels, w_hat), key=lambda t: -abs(t[1]))[:6]]
    print("\n坐标搜索（前 6 个）")
    w = dict(best[0])
    for lab in order:
        cur = w[lab]
        for sc in (0.0, 0.5, 0.75, 1.25, 1.5, 2.0):
            if nei >= a.max_eval:
                break
            cand = dict(w)
            cand[lab] = cur * sc
            m = evaluate(fitH, fitR, cand, FIT)
            nei += 1
            if J(m) > best[1]:
                best = (cand, J(m), m)
                w = cand
        print("   %-8s → %.4f" % (lab, w[lab]))
    print("评估次数 %d" % nei)

    wbest = best[0]
    mb = best[2]
    vb = evaluate(valH, valR, wbest, VAL)
    print("\n== 最优解 ==")
    for lab in labels:
        if abs(wbest[lab]) > 1e-9:
            print("   %-8s δ %+.4f → 有效权重 %+.4f" % (lab, wbest[lab], W1363.get(lab, 0.0) + wbest[lab]))
    print("fit 1363+δ  IC %.5f IR %.4f 压力 %.4f 换手 %.4f（J %.4f）" % (mb["ic_mean"], mb["ic_ir"], mb["stress"], mb["turnover"], J(mb)))
    print("val 1363+δ  IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (vb["ic_mean"], vb["ic_ir"], vb["stress"], vb["turnover"]))
    print("val 相对 1363：IC %+.5f IR %+.4f 压力 %+.4f 换手 %+.4f" % (vb["ic_mean"] - v0["ic_mean"], vb["ic_ir"] - v0["ic_ir"], vb["stress"] - v0["stress"], vb["turnover"] - v0["turnover"]))
    print("val 相对贪心：IC %+.5f IR %+.4f 压力 %+.4f 换手 %+.4f" % (vb["ic_mean"] - g2["ic_mean"], vb["ic_ir"] - g2["ic_ir"], vb["stress"] - g2["stress"], vb["turnover"] - g2["turnover"]))


if __name__ == "__main__":
    main()
