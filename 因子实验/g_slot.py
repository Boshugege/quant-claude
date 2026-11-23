# -*- coding: utf-8 -*-
"""G3：分槽位权重（七个截面分别给第二头部 s155 一个权重）。

B = 1363原始值 + Σ_slot w_slot · rank(s155)·1[该截面]。拟合窗 2020-2022（J = IC_IR + 压力 − λ·换手），
验证窗 2023-2024。用 g_fast 的精确快速口径（逐分量 residual，再由 sd 加权求和）。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_eval as D  # noqa: E402
import d2_optimize as O  # noqa: E402
from g_fast import Fast, _en, WINS, ALL  # noqa: E402

SLOTS = [1000, 1030, 1100, 1130, 1330, 1400, 1430]
HEADS = {"1363": ["1015-203626-f5f0", "1017-102020-9a7c", "1017-102056-e592"],
         "s155": ["fm_s155", "o21_s155", "o20_s155"]}
FITW = ("2020-01-01", "2022-12-31")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="s155")
    ap.add_argument("--lam", type=float, default=0.2)
    ap.add_argument("--max-eval", type=int, default=60)
    ap.add_argument("--base-w", type=float, default=0.25)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    F = Fast(None)
    c = F.c
    heads = {}
    for hn, tm in HEADS.items():
        arrs = []
        for i in range(3):
            x = R.load(tm[i])[0]
            x[c.slot_of_date[c.date_id] == 930] = 0.0
            arrs.append(x)
        heads[hn] = arrs
        for i in range(3):
            F.comp(hn + "#" + str(i), arrs[i])
    F.heads = heads
    print("底座分量就绪")

    # 每窗口每槽位的分量
    for i, rng in enumerate([("2020-01-01", "2024-12-31"), FITW, FITW]):
        nm = {0: "fm_", 1: "o21_", 2: "o20_"}[i] + a.root
        ax = O.axis(c, nm, *rng)
        for s in SLOTS:
            z = ax.copy()
            z[c.slot_of_date[c.date_id] != s] = 0.0
            F.comp("%s@%d#%d" % (a.root, s, i), z)
        print("  窗口 %d 的 %d 个槽位分量就绪" % (i, len(SLOTS)))

    def mix(w_slot, i):
        v = {"1363#" + str(i): 1.0}
        for s in SLOTS:
            k = "%s@%d#%d" % (a.root, s, i)
            if k in F.P:
                v[k] = w_slot.get(s, 0.0)
        return F.mix(v, heads["1363"][i])

    cache = {}

    def ser(w_slot, i):
        key = json.dumps({str(k): round(v, 6) for k, v in sorted(w_slot.items())}, sort_keys=True) + "#" + str(i)
        if key not in cache:
            cache[key] = D.series_from_res(mix(w_slot, i), *ALL)
        return cache[key]

    def score(w_slot, i, wn):
        s = ser(w_slot, i)
        return c.summarize(s, (s["day"] >= pd.Timestamp([q for q in WINS if q[0] == wn][0][1]).value)
                          & (s["day"] <= _en([q for q in WINS if q[0] == wn][0][2]).value))

    def fit_J(w_slot):
        m = c.summarize(ser(w_slot, 1), (ser(w_slot, 1)["day"] >= pd.Timestamp(FITW[0]).value)
                        & (ser(w_slot, 1)["day"] <= _en(FITW[1]).value))
        return m["ic_ir"] + m["stress"] - a.lam * m["turnover"], m

    base_w = {s: a.base_w for s in SLOTS}
    J0, m0 = fit_J(base_w)
    print("固定权重 %.3f：fit  J %.4f IC %.5f IR %.4f 压力 %.4f 换手 %.4f"
          % (a.base_w, J0, m0["ic_mean"], m0["ic_ir"], m0["stress"], m0["turnover"]))
    best = (dict(base_w), J0, m0)
    w = dict(base_w)
    nv = 0
    for it in range(3):
        for s in SLOTS:
            cur = w[s]
            for sc in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                if nv >= a.max_eval:
                    break
                cand = dict(w)
                cand[s] = cur * sc
                J, m = fit_J(cand)
                nv += 1
                if J > best[1]:
                    best = (cand, J, m)
                    w = cand
        print("  第 %d 轮：%s" % (it + 1, {s: round(v, 3) for s, v in w.items()}))
    print("评估次数 %d" % nv)
    print("\n== 分槽位最优 ==  %s" % json.dumps({str(s): round(v, 4) for s, v in best[0].items()}, ensure_ascii=False))
    print("fit  J %.4f IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (best[1], best[2]["ic_mean"], best[2]["ic_ir"], best[2]["stress"], best[2]["turnover"]))
    for wn in ("2023", "2024"):
        i = 0
        mb = score(base_w, i, wn)
        mf = score(best[0], i, wn)
        print("  %s 固定(%.2f) IC %.5f IR %.4f 压力 %.4f 换手 %.4f | 分槽位 IC %.5f IR %.4f 压力 %.4f 换手 %.4f | ΔIC %+.5f ΔIR %+.4f Δ换手 %+.4f"
              % (wn, a.base_w, mb["ic_mean"], mb["ic_ir"], mb["stress"], mb["turnover"],
                 mf["ic_mean"], mf["ic_ir"], mf["stress"], mf["turnover"],
                 mf["ic_mean"] - mb["ic_mean"], mf["ic_ir"] - mb["ic_ir"], mf["turnover"] - mb["turnover"]))
    print("\n分槽位 vs 1363（五窗口）：")
    TOL = {"ic_mean": 2e-4, "ic_ir": 5e-3, "stress": 5e-3, "turnover": 0.0}
    for wn, _, _ in WINS:
        i = 0 if wn in ("2023", "2024") else (1 if wn in ("2021", "2022") else 2)
        mb = score({}, i, wn)
        mf = score(best[0], i, wn)
        ok = all((mf[k] - mb[k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in TOL)
        print("  %s ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f %s"
              % (wn, mf["ic_mean"] - mb["ic_mean"], mf["ic_ir"] - mb["ic_ir"],
                 mf["stress"] - mb["stress"], mf["turnover"] - mb["turnover"], "★" if ok else ""))
    if a.out:
        json.dump({"slots": SLOTS, "w": {str(s): best[0][s] for s in SLOTS}, "lam": a.lam,
                   "fit": best[2]}, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("写出", a.out)


if __name__ == "__main__":
    main()
