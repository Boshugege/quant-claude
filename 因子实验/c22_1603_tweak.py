# -*- coding: utf-8 -*-
"""1603（公榜 0.87340）的单分量微调：找五个窗口五项都不差的升级（平台口径：去 Y_sd5、X_n5d，N_dir 活）。
微调对象：N_dir（确认活着、降换手）、Z_mfr60、W_zero、k1406、s161 头部，以及与小锚对的组合。Δ 相对 1603 平台口径。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = ("2023", "2024", "2021", "2022", "2020")
U.DEAL = {"Y_sd5", "X_n5d"}
base = {ix: build("1603", ix, drop_deal=True) for ix in (0, 1, 2)}
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn in W}
ref = ev5(base)
K = 0.655 / 0.289
VARS = [("N_dir", -0.05, 0, 0), ("N_dir", -0.10, 0, 0), ("Z_mfr60", -0.05, 0, 0), ("W_zero", 0.05, 0, 0),
        ("k1406", 0.03, 0, 0), ("s161", 0.10, 0, 0), ("s161", -0.10, 0, 0),
        ("N_dir", -0.05, 0.02, 0.03), ("N_dir", -0.10, 0.02, 0.03), ("s161", 0.10, 0.02, 0.03)]
for lab, dw, b, e in VARS:
    Bs = {}
    for ix in (0, 1, 2):
        x = base[ix] + dw * G.axis(lab, ix) + b * K * CLS * (slot == 1430) + e * K * CP
        x[z] = 0.0
        Bs[ix] = x
    r = ev5(Bs); line, star = "1603 %s%+.2f CLS1430+%.2f CP%.2f" % (lab, dw, b, e), 0
    for wn in W:
        d = {m: r[wn][m] - ref[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
