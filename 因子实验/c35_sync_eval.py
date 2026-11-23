# -*- coding: utf-8 -*-
"""E2 评估：SYNC20（价格同步性）单独五窗口（正、反两个方向），以及按锚的方式叠到 1610（平台口径）上：1610 + w·K10·秩(±SYNC20)。"""
import numpy as np
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
def rk(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
    x[x == 0] = np.nan
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
    r[z] = 0.0
    return r
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
S = rk("SYNC20")
for nm, x in (("+SYNC20", S), ("-SYNC20", -S)):
    print("单独 %s " % nm + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[G.ev(x, wn)[m] for m in MET]) for wn in W), flush=True)
base = {wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in W.items()}
ref = {wn: G.ev(base[wn], wn) for wn in W}
for sg in (1, -1):
    for w in (0.025, 0.05, 0.075):
        line, st = "1610 + %+.3f·SYNC20" % (sg * w), 0
        for wn in W:
            r = G.ev(base[wn] + sg * w * K10 * S, wn)
            d = {m: r[m] - ref[wn][m] for m in MET}
            st += ok(d)
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
        print(line + "  %d/5" % st, flush=True)
