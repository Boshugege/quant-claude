# -*- coding: utf-8 -*-
"""树的多种子／多配置平均（原始分取平均 = 把几个模型的树拼起来、学习率减半，可直接做成一个代码版），与 1600 平台口径比五窗口五项。"""
import sys
import numpy as np
from g_go import GO
from c1_upgrades import OUT
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
W = ("2023", "2024", "2021", "2022", "2020")
G = GO(); c = G.c; z = c.slot_of_date[c.date_id] == 930
L = lambda t, wn: np.where(z, 0.0, np.load(OUT / ("h2_%s_%s.npy" % (t, wn))))
ref = {wn: G.ev(L("seasplat", wn), wn) for wn in W}
COMBOS = [x.split("+") for x in sys.argv[1:]]
for tags in COMBOS:
    line, star = "avg(%s)" % "+".join(tags), 0
    for wn in W:
        F = np.mean([L(t, wn) for t in tags], axis=0)
        r = G.ev(F, wn)
        d = {m: r[m] - ref[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
