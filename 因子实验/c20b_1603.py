# -*- coding: utf-8 -*-
"""1601／1603 再加锚的细网格：要求两种口径（去 Y+X、三个都去）都 5/5。Δ 相对各自（平台口径）。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = ("2023", "2024", "2021", "2022", "2020")
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn in W}
GRID = {
        "1603": ((0.005, 0.01, 0.02), (0.01, 0.02, 0.03), (0.0, 0.02, 0.03), (0.0, 0.03, 0.04), (0.005, 0.02, 0.025))}
SD = {"1601": 0.544, "1603": 0.655}
res = {}
for dname, dset in (("去Y+X", {"Y_sd5", "X_n5d"}),):
    U.DEAL = dset
    for kind, vs in GRID.items():
        K = SD[kind] / 0.289
        base = {ix: build(kind, ix, drop_deal=True) for ix in (0, 1, 2)}
        ref = ev5(base)
        for a, b, e in vs:
            Bs = {}
            for ix in (0, 1, 2):
                x = base[ix] + a * K * CLS + b * K * CLS * (slot == 1430) + e * K * CP
                x[z] = 0.0
                Bs[ix] = x
            r = ev5(Bs); line, star = "%s %s +CLS%.3f+%.3f CP%.3f" % (kind, dname, a, b, e), 0
            for wn in W:
                d = {m: r[wn][m] - ref[wn][m] for m in MET}
                ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
                star += ok
                line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
            print(line + "  %d/5" % star, flush=True)
