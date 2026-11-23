# -*- coding: utf-8 -*-
"""因子头脑风暴第二步：评估 bs_<名>.npy。用法：c30_eval.py <分片号> <分片数> [因子名…]
每个因子：单独（秩）五窗口 IC／IC_IR／换手；叠到 1610、1611（平台口径）上 ±0.03／±0.06（秩口径 × 底座 k），报五窗口 ★ 数与最好那组的 Δ。"""
import sys
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, W1363, MET, OUT
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = ("2023", "2024", "2021", "2022", "2020")
ALL = ["r_last", "cvwap", "r_day", "r_intra", "pos", "obi1", "obi3", "spread", "ocnt", "osize", "depth", "obi_chg", "tsize_rel",
       "ushadow", "lshadow", "amihud", "pv_corr", "vol_surge", "rng_rel", "rev_x_vol", "r_mkt", "gap", "p_tail", "p_cvw"]
sh, nsh = int(sys.argv[1]), int(sys.argv[2])
names = sys.argv[3:] or [f for i, f in enumerate(ALL) if i % nsh == sh]
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn in W}
def b1379(ix):
    B = np.asarray(G.B[("1363", ix)], float).copy()
    for r in W1363:
        B += (0.5 * W1363[r] - (1.5 * W1363[r] if r in U.DEAL else 0.0)) * G.axis(r, ix)
    return B
K10, K11 = 0.2774 / 0.289, 0.50 / 0.289
BASES = {"1610": ({ix: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for ix in (0, 1, 2)}, K10),
         "1611": ({ix: np.where(z, 0.0, b1379(ix) + 0.05 * K11 * CLS + 0.10 * K11 * CLS * (slot == 1430) + 0.075 * K11 * CP) for ix in (0, 1, 2)}, K11)}
REF = {k: ev5(v[0]) for k, v in BASES.items()}
for f in names:
    x = np.load(OUT / ("bs_%s.npy" % f))
    r = np.nan_to_num(U.O.rc_of(x, c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    so = ev5({0: r, 1: r, 2: r})
    print("[%s] 单独 %s" % (f, " | ".join("%s %+.4f/%+.3f/%.3f" % (wn, so[wn]["ic_mean"], so[wn]["ic_ir"], so[wn]["turnover"]) for wn in W)), flush=True)
    for bn, (Bs, K) in BASES.items():
        best = None
        for w in (0.03, -0.03, 0.06, -0.06):
            res = ev5({ix: np.where(z, 0.0, Bs[ix] + w * K * r) for ix in (0, 1, 2)})
            star, line = 0, ""
            for wn in W:
                d = {m: res[wn][m] - REF[bn][wn][m] for m in MET}
                ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
                star += ok
                line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
            score = star + 0.001 * sum(res[wn]["ic_ir"] - REF[bn][wn]["ic_ir"] for wn in W)
            if best is None or score > best[0]:
                best = (score, w, star, line)
        print("[%s] 叠到 %s 最好 w=%+.2f %d/5%s" % (f, bn, best[1], best[2], best[3]), flush=True)
