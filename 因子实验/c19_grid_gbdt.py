# -*- coding: utf-8 -*-
"""(1) 1611 小权重网格：1379 + CLS／CP，要求两种口径（去 Y+X、三个都去）都 5/5；
(2) 树方向：1607（c2pf）相对 1600（seas，含成交笔数特征的本地版）的五窗口五项；以及 c2pf_aux05 相对 1607。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, W1363, MET, OUT
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = ("2023", "2024", "2021", "2022", "2020")
def evd(F):
    return {wn: G.ev(F[wn], wn) for wn in W}
def rep(name, ref, r):
    line, star = name, 0
    for wn in W:
        d = {m: r[wn][m] - ref[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
def b1379(ix, dset):
    B = np.asarray(G.B[("1363", ix)], float).copy()
    for r in W1363:
        B += (0.5 * W1363[r] - (1.5 * W1363[r] if r in dset else 0.0)) * G.axis(r, ix)
    B[z] = 0.0
    return B
K = 0.50 / 0.289
for dname, dset in (("去Y+X", {"Y_sd5", "X_n5d"}), ("三个都去", {"N_dir", "Y_sd5", "X_n5d"})):
    base = {ix: b1379(ix, dset) for ix in (0, 1, 2)}
    ref = evd({wn: base[IXW[wn]] for wn in W})
    for a, b, e in ((0.02, 0.04, 0.03), (0.03, 0.06, 0.03), (0.02, 0.04, 0.05), (0.03, 0.03, 0.04), (0.04, 0.04, 0.05), (0.03, 0.06, 0.04)):
        F = {}
        for wn in W:
            x = base[IXW[wn]] + a * K * CLS + b * K * CLS * (slot == 1430) + e * K * CP
            x[z] = 0.0
            F[wn] = x
        rep("1379 %s +CLS%.2f+%.2f CP%.2f" % (dname, a, b, e), ref, evd(F))
def load(tag):
    return {wn: np.where(z, 0.0, np.load(OUT / ("h2_%s_%s.npy" % (tag, wn)))) for wn in W}
S, P, A = evd(load("seas")), evd(load("c2pf")), evd(load("c2pf_aux05"))
print("1600(seas) " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[S[wn][m] for m in MET]) for wn in W), flush=True)
print("1607(c2pf) " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[P[wn][m] for m in MET]) for wn in W), flush=True)
rep("1607 vs 1600(本地全含)", S, P)
rep("aux05 vs 1607", P, A)
