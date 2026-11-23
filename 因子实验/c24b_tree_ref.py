# -*- coding: utf-8 -*-
"""树的多模型平均，同时对照 1600 平台口径（seasplat）与 1617（深度 4 ×8 + 深度 5 ×8），五窗口五项；★ = 该窗口五项都不差（IC 容差 −0.00005）。
用法：c24b_tree_ref.py 组合1 组合2 …（组合 = 用 + 连接的标签）"""
import sys
import numpy as np
from g_go import GO
from c1_upgrades import OUT
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
W = ("2023", "2024", "2021", "2022", "2020")
D4 = "d4t150a025s+t150s11+t150s23+t150s42+d4t130a025s+d4t130a025s_s11+d4t130a025s_s23+t130s42"
D5 = "d5a040s+d5a040s_s11+d5a040s_s23+d5a040s_s42+d5a040s_s3+d5a040s_s5+d5a040s_s13+d5a040s_s17"
G = GO(); c = G.c; z = c.slot_of_date[c.date_id] == 930
L = lambda t, wn: np.where(z, 0.0, np.load(OUT / ("h2_%s_%s.npy" % (t, wn))))
avg = lambda tags, wn: np.mean([L(t, wn) for t in tags], axis=0)
REF = {"1600": {wn: G.ev(L("seasplat", wn), wn) for wn in W}, "1617": {wn: G.ev(avg((D4 + "+" + D5).split("+"), wn), wn) for wn in W}}
import os
for x in filter(None, os.environ.get("REFS", "").split(",")):              # 额外的对照：REFS=名字=组合,…
    nm, cb = x.split("=", 1)
    REF[nm] = {wn: G.ev(avg(cb.replace("D4", D4).replace("D5", D5).split("+"), wn), wn) for wn in W}
for combo in sys.argv[1:]:
    tags = combo.replace("D4", D4).replace("D5", D5).split("+")
    R = {wn: G.ev(avg(tags, wn), wn) for wn in W}
    for rn, ref in REF.items():
        line, star = "%-60s vs %s" % (combo[:60], rn), 0
        for wn in W:
            d = {m: R[wn][m] - ref[wn][m] for m in MET}
            ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            star += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % star, flush=True)
