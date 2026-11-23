# -*- coding: utf-8 -*-
"""复活拆开看：只复活 Y_sd5 或只复活 X_n5d（平台口径 + s×(只活一根 − 平台)），s = 0.5／1.0；1603 与 1612，五窗口五项。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CLS, CP, build, MET
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K12 = 0.544 / 0.289
def mk(kind, ix, dead):
    U.DEAL = set(dead)
    x = build(kind, ix, drop_deal=bool(dead))
    if kind == "1601":
        x = x + 0.01 * K12 * CLS + 0.02 * K12 * CLS * (slot == 1430) + 0.03 * K12 * CP
    return np.where(z, 0.0, x)
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
for nm, kind in (("1603", "1603"), ("1612", "1601")):
    P = {ix: mk(kind, ix, ("Y_sd5", "X_n5d")) for ix in (0, 1, 2)}
    ref = {wn: G.ev(P[ix], wn) for wn, ix in W.items()}
    for alive, dead in (("Y_sd5", ("X_n5d",)), ("X_n5d", ("Y_sd5",))):
        Dd = {ix: mk(kind, ix, dead) - P[ix] for ix in (0, 1, 2)}
        for sc in (0.5, 1.0):
            line, st = "%s 只活 %s ×%.1f" % (nm, alive, sc), 0
            for wn, ix in W.items():
                q = G.ev(P[ix] + sc * Dd[ix], wn); d = {m: q[m] - ref[wn][m] for m in MET}
                st += ok(d)
                line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
            print(line + "  %d/5" % st, flush=True)
