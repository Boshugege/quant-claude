# -*- coding: utf-8 -*-
"""E5（10-19）：1610 的逐分量权重微调（1610 的根权重是从 1363 借来的，从没按 161 头部重新调过）。
每个分量 k 加 δ·K10·分量k，δ ∈ {±0.05, ±0.10}。只在 2021／2022／2020 上挑（选择窗），2023／2024 只作检验（留出窗）。"""
import sys
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT, ROOTS
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2021": 1, "2022": 1, "2020": 2, "2023": 0, "2024": 0}
SEL, HOLD = ("2021", "2022", "2020"), ("2023", "2024")
K10 = 0.2774 / 0.289
yr = pd.DatetimeIndex(c.day_of_date[c.date_id]).year
fit = (yr <= 2022) & (~z)
def zreg(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
    mu, sd = x[fit].mean(), x[fit].std()
    r = np.clip((x - mu) / sd, -3, 3)
    r[z] = 0.0
    return r
def rk(x):
    r = np.nan_to_num(O.rc_of(np.asarray(x, float), c.date_id, c.n_dates), nan=0.0)
    r[z] = 0.0
    return r
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
base = {wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in W.items()}
ref = {wn: G.ev(base[wn], wn) for wn in W}
print("1610 底座 " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
comps = {}
for wn, ix in W.items():
    cc = {"s161": rk(G.B[("s161", ix)]), "CLS": CLS, "CP": CP}
    for r in ROOTS:
        if r not in U.DEAL:
            cc[r] = G.axis(r, ix)
    comps[wn] = cc
for k in comps["2023"]:
    for dl in (-0.10, -0.05, 0.05, 0.10):
        line, s_sel, s_hold = "%-8s δ%+.2f" % (k, dl), 0, 0
        for wn in W:
            r = G.ev(base[wn] + dl * K10 * comps[wn][k], wn)
            d = {m: r[m] - ref[wn][m] for m in MET}
            o = ok(d)
            s_sel += o and wn in SEL
            s_hold += o and wn in HOLD
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if o else "")
        print(line + "  选择 %d/3 留出 %d/2" % (s_sel, s_hold), flush=True)
