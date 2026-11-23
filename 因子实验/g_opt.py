# -*- coding: utf-8 -*-
"""通用定权器（第十三轮 G1/G4）：B = 底座原始值 + Σ_j δ_j·rank_j。

拟合窗 2020-01～2022-12，验证窗 2023-01～2024-12；目标 J = IC_IR + 压力 − λ·换手，
只在拟合窗上搜。方向由逐截面岭回归给出（样本用拟合窗掩码），再在 1 维尺度 + 前 8 个坐标上做局部搜索。
`--penalties` 可一次跑多档 λ（复用轴与岭回归方向）；`--warm` 用上一轮的 δ 当额外起点。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402

# g_opt 专用候选（G4 换手前沿）：1391 慢特征 + 1400 族里换手≈0 的日频根
EXTRA = [("r1391", "fm_1391", "o21_1391", "o20_1391", 0.0),
         ("r1406", "fm_1406", "o21_1406", "o20_1406", 0.0),
         ("r1407", "fm_1407", "o21_1407", "o20_1407", 0.0),
         ("r1409", "fm_1409", "o21_1409", "o20_1409", 0.0)]

FIT = ("2020-01-01", "2022-12-31")
VAL = ("2023-01-01", "2024-12-31")


def _span(span):
    en = pd.Timestamp(span[1]).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return span[0], en


def base_pair(main, o21, o20):
    c = R.core()
    f = np.full(c.n_rows, np.nan)
    for nm in (o21, o20):
        if nm:
            g = R.load(nm)[0]
            f = np.where(np.isfinite(f), f, g)
    v = R.load(main)[0] if main else np.full(c.n_rows, np.nan)
    z930 = c.slot_of_date[c.date_id] == 930
    f[z930] = 0.0
    v[z930] = 0.0
    return f, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-main", required=True)
    ap.add_argument("--base-o21", default="")
    ap.add_argument("--base-o20", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--penalties", default="0.2")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--max-eval", type=int, default=80)
    ap.add_argument("--gammas", default="0.3,0.5,0.75,1.0,1.5,2.0,3.0")
    ap.add_argument("--warm", default="", help="上一轮 g_opt 写出的 JSON（用它的 delta 当额外起点）")
    ap.add_argument("--out", default="")
    ap.add_argument("--aux-w", type=float, default=0.0, help="E3（10-19）：岭回归方向的目标 = 秩(当前标签) + w·秩(同日之后 --aux-n 个截面标签之和)；局部搜索的目标不变")
    ap.add_argument("--aux-n", type=int, default=2)
    ap.add_argument("--sub", default="", help="10-18：底座先减去这些根（平台上为零），如 Y_sd5:0.10,X_n5d:-0.10（权重 × 根轴）")
    a = ap.parse_args()

    c = R.core()
    nd = c.n_dates
    allc = list(O.CAND) + EXTRA
    labs = [x for x in a.labels.split(",") if x] or [l for l, *_ in allc]
    spec = {l: (n1, n21, n20, al) for l, n1, n21, n20, al in allc}
    base_f, base_v = base_pair(a.base_main, a.base_o21, a.base_o20)
    for it in [x for x in a.sub.split(",") if x]:                      # 平台口径：底座里的死根置零
        l, w = it.split(":")
        n1, n21, n20, al = spec[l]
        gf = np.full(c.n_rows, np.nan)
        for nm in (n21, n20):
            if nm:
                g0 = O.axis(c, nm, *FIT, alpha=al)
                gf = np.where(np.isfinite(gf), gf, g0)
        base_f = base_f - float(w) * np.nan_to_num(gf, nan=0.0)
        base_v = base_v - float(w) * np.nan_to_num(O.axis(c, n1, *VAL, alpha=al), nan=0.0)
        print("底座减去 %s × %s" % (w, l), flush=True)
    print("底座 %s：fit 有效 %d ; val 有效 %d" % (a.base_main, np.isfinite(base_f).sum(), np.isfinite(base_v).sum()))

    A_fit, A_val = {}, {}
    for l in labs:
        n1, n21, n20, al = spec[l]
        f = np.full(c.n_rows, np.nan)
        for nm in (n21, n20):
            if nm:
                g = O.axis(c, nm, *FIT, alpha=al)
                f = np.where(np.isfinite(f), f, g)
        A_fit[l] = np.where(c.slot_of_date[c.date_id] == 930, 0.0, np.nan_to_num(f, nan=0.0)) if False else f
        A_val[l] = O.axis(c, n1, *VAL, alpha=al)
    print("轴构造完成（%d 个）" % len(labs))

    def ev(B, span):
        st, en = _span(span)
        m, _ = c.evaluate(B, start=st, end=en)
        return m

    def build(d, which):
        B = np.asarray(base_f if which == "fit" else base_v, float).copy()
        Ax = A_fit if which == "fit" else A_val
        for l, val in d.items():
            if val:
                B = B + val * np.nan_to_num(Ax[l], nan=0.0)
        return B

    f0 = ev(base_f, FIT)
    v0 = ev(base_v, VAL)
    print("fit 底座     IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (f0["ic_mean"], f0["ic_ir"], f0["stress"], f0["turnover"]))
    print("val 底座     IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (v0["ic_mean"], v0["ic_ir"], v0["stress"], v0["turnover"]))

    # ---- 岭回归方向（只用拟合窗）
    _ok, did, slot = R.base_rows(*FIT)
    idx = np.flatnonzero(_ok)
    g = did[idx]
    X = np.column_stack([np.nan_to_num(np.asarray(A_fit[l][idx], float)) for l in labs])
    ry = O.rc_of(c.y[idx], g, nd)
    if a.aux_w > 0:
        dfa = pd.DataFrame({"inst": c.inst, "sec": c.date_id, "day": c.day_of_date[c.date_id], "y": c.y}).sort_values(["inst", "sec"])
        ga = dfa.groupby(["inst", "day"], sort=False)["y"]
        nxt = sum(ga.shift(-k) for k in range(1, a.aux_n + 1))
        ysum = np.full(c.n_rows, np.nan)
        ysum[dfa.index.to_numpy()] = nxt.to_numpy()
        ry = ry + a.aux_w * np.nan_to_num(O.rc_of(ysum[idx], g, nd), nan=0.0)
        print("岭回归目标加多期：w=%.2f n=%d" % (a.aux_w, a.aux_n), flush=True)
    good = np.isfinite(ry)
    idx, g, X, ry = idx[good], g[good], X[good], ry[good]
    cnt = np.maximum(np.bincount(g, minlength=nd), 1).astype(float)[g]
    for j in range(X.shape[1]):
        X[:, j] -= np.bincount(g, X[:, j], nd)[g] / cnt
    ry = ry - np.bincount(g, ry, nd)[g] / cnt
    w_hat = np.linalg.solve(X.T @ X + a.ridge * np.eye(X.shape[1]), X.T @ ry)
    print("\n岭回归方向（样本 %d 行）" % len(idx))
    for l, wv in sorted(zip(labs, w_hat), key=lambda t: -abs(t[1]))[:12]:
        print("   %-8s %+.4f" % (l, wv))
    del X, ry

    w_warm = {}
    if a.warm:
        j = json.load(open(a.warm, encoding="utf-8"))
        w_warm = {l: float(j["delta"].get(l, 0.0)) for l in labs}
        print("warm start：%s" % json.dumps({k: round(v, 4) for k, v in w_warm.items() if abs(v) > 1e-9}, ensure_ascii=False))

    order = [l for l, _ in sorted(zip(labs, w_hat), key=lambda t: -abs(t[1]))[:8]]
    res = {}
    for pen in [float(x) for x in a.penalties.split(",")]:
        def J(m):
            return m["ic_ir"] + m["stress"] - pen * m["turnover"]
        best = ({l: 0.0 for l in labs}, J(f0), f0)
        if w_warm:
            m = ev(build(w_warm, "fit"), FIT)
            if J(m) > best[1]:
                best = (dict(w_warm), J(m), m)
        print("\n=== λ = %.2f ===" % pen)
        for gam in [float(x) for x in a.gammas.split(",")]:
            w = {l: gam * wv for l, wv in zip(labs, w_hat)}
            m = ev(build(w, "fit"), FIT)
            if J(m) > best[1]:
                best = (w, J(m), m)
            print("   γ=%.2f J %.4f IC %.5f IR %.4f 压力 %.4f 换手 %.4f"
                  % (gam, J(m), m["ic_mean"], m["ic_ir"], m["stress"], m["turnover"]))
        nv = 0
        w = dict(best[0])
        for l in order:
            cur = w[l]
            for sc in (0.0, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0):
                if nv >= a.max_eval:
                    break
                cand = dict(w)
                cand[l] = cur * sc
                m = ev(build(cand, "fit"), FIT)
                nv += 1
                if J(m) > best[1]:
                    best = (cand, J(m), m)
                    w = cand
        wbest = best[0]
        mb, vb = best[2], ev(build(wbest, "val"), VAL)
        print("   最优 δ = %s" % json.dumps({k: round(v, 4) for k, v in wbest.items() if abs(v) > 1e-9}, ensure_ascii=False))
        print("   fit  J %.4f IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (J(mb), mb["ic_mean"], mb["ic_ir"], mb["stress"], mb["turnover"]))
        print("   val  IC %.5f IR %.4f LS %.2f 压力 %.4f 换手 %.4f" % (vb["ic_mean"], vb["ic_ir"], vb["ls_sharpe"], vb["stress"], vb["turnover"]))
        print("   val 相对底座 ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f"
              % (vb["ic_mean"] - v0["ic_mean"], vb["ic_ir"] - v0["ic_ir"], vb["stress"] - v0["stress"], vb["turnover"] - v0["turnover"]))
        for st, en in (("2023-01-01", "2023-12-31"), ("2024-01-01", "2024-12-31")):
            m, b = ev(build(wbest, "val"), (st, en)), ev(base_v, (st, en))
            print("   %s ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f"
                  % (st[:4], m["ic_mean"] - b["ic_mean"], m["ic_ir"] - b["ic_ir"], m["stress"] - b["stress"], m["turnover"] - b["turnover"]))
        for st, en in (("2020-01-01", "2020-12-31"), ("2021-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31")):
            m, b = ev(build(wbest, "fit"), (st, en)), ev(base_f, (st, en))
            print("   %s ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f"
                  % (st[:4], m["ic_mean"] - b["ic_mean"], m["ic_ir"] - b["ic_ir"], m["stress"] - b["stress"], m["turnover"] - b["turnover"]))
        res["%.4f" % pen] = {"penalty": pen, "delta": wbest, "fit": mb, "val": vb, "val_base": v0, "fit_base": f0}
    if a.out:
        json.dump({"labels": labs, "runs": res}, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("\n写出", a.out)


if __name__ == "__main__":
    main()
