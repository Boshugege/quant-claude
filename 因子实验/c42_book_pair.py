# -*- coding: utf-8 -*-
"""下午 B3（10-19）：盘口根成对叠加。消耗率 BK_CONS（反向）与恢复 BK_RESIL（反向）在 1603 上各自 3/5 且窗口互补；
再加日内不变的 BK_AOS_20（正向）。底座 + K·(−a·秩CONS − b·秩RESIL + c·秩AOS20)，a, b ∈ {0, 0.0125, 0.025}，c ∈ {0, 0.025}；1603 与 1610，五窗口五项。"""
import itertools
import numpy as np
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10, K03 = 0.2774 / 0.289, 0.655 / 0.289
def rk(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float); x[x == 0] = np.nan
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    return r
CONS, RES, AOS = rk("BK_CONS"), rk("BK_RESIL"), rk("BK_AOS_20")
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
BASES = {"1603": ({wn: np.where(z, 0.0, build("1603", ix, drop_deal=True)) for wn, ix in W.items()}, K03),
         "1610": ({wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in W.items()}, K10)}
for b, (B, K) in BASES.items():
    ref = {wn: G.ev(B[wn], wn) for wn in W}
    for a1, b1, c1 in itertools.product((0, 0.0125, 0.025), (0, 0.0125, 0.025), (0, 0.025)):
        if a1 == b1 == c1 == 0:
            continue
        line, st = "%s CONS-%.4f RESIL-%.4f AOS20+%.3f" % (b, a1, b1, c1), 0
        for wn in W:
            r = G.ev(B[wn] + K * (-a1 * CONS - b1 * RES + c1 * AOS), wn)
            d = {m: r[m] - ref[wn][m] for m in MET}
            st += ok(d)
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
        print(line + "  %d/5" % st, flush=True)
