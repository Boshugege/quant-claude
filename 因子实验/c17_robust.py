# -*- coding: utf-8 -*-
"""F1 稳健性 + F5：
 (1) 1610 候选在"三个成交笔数根都死"的底座上是否仍 5/5；
 (2) 1601／1603（公榜已知）加 CLOSEPOS20 锚是否五项都不差（底座口径：去 Y+X；三个都去）。Δ 相对各自底座。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = ("2023", "2024", "2021", "2022", "2020")
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn in W}
def rep(name, ref, Bs):
    r = ev5(Bs); line, star = name, 0
    for wn in W:
        d = {m: r[wn][m] - ref[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
SD = {"1602": 0.2774, "1601": 0.544, "1603": 0.655}
for dname, dset in (("去Y+X", {"Y_sd5", "X_n5d"}), ("三个都去", {"N_dir", "Y_sd5", "X_n5d"})):
    U.DEAL = dset
    for kind in ("1602", "1601", "1603"):
        K = SD[kind] / 0.289
        base = {ix: build(kind, ix, drop_deal=True) for ix in (0, 1, 2)}
        ref = ev5(base)
        print("== %s %s 底座 " % (kind, dname) + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
        if kind == "1602":
            vs = ((0.05, 0.10, 0.075), (0.03, 0.06, 0.05), (0.05, 0.10, 0.10))
        else:
            vs = ((0, 0, 0.03), (0, 0, 0.05), (0, 0, 0.075))
        for a, b, e in vs:
            Bs = {}
            for ix in (0, 1, 2):
                x = base[ix] + a * K * CLS + b * K * CLS * (slot == 1430) + e * K * CP
                x[z] = 0.0
                Bs[ix] = x
            rep("%s %s CLS%.2f+%.2f CP%.3f" % (kind, dname, a, b, e), ref, Bs)
