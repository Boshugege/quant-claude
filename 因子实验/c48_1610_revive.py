# -*- coding: utf-8 -*-
"""1610 方向也复活 Y_sd5／X_n5d：1610（平台口径，去两根）+ s×(带两根 − 不带)，s = 0.5／1.0，五窗口五项（2023／2024 为样本外重点）。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CLS, CP, build, MET
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
def b(ix, full):
    U.DEAL = set() if full else {"Y_sd5", "X_n5d"}
    return np.where(z, 0.0, build("1602", ix, drop_deal=not full) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP)
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
P = {ix: b(ix, False) for ix in (0, 1, 2)}
D = {ix: b(ix, True) - P[ix] for ix in (0, 1, 2)}
ref = {wn: G.ev(P[ix], wn) for wn, ix in W.items()}
for sc in (0.25, 0.5, 1.0):
    line, st = "1610 + %.2f×复活" % sc, 0
    for wn, ix in W.items():
        q = G.ev(P[ix] + sc * D[ix], wn); d = {m: q[m] - ref[wn][m] for m in MET}
        st += ok(d)
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
    print(line + "  %d/5" % st, flush=True)
