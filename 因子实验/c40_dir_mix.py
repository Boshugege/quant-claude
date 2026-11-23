# -*- coding: utf-8 -*-
"""E8（10-19）：两个公榜方向 1617（树）与 1610（161 头部线性）之间的混合。
(a) 逐截面秩混合 (1−w)·秩(1617) + w·秩(1610)，w = 0.2／0.35／0.5：看 IC_IR 有没有超过两者（分散化红利），以及相对两端的位置。
(b) 分时段选方向：每个时段在 2020–22（2021／2022／2020 三窗）上 IC_IR 平均更高的方向，2023／2024 检验。"""
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
D4 = "d4t150a025s+t150s11+t150s23+t150s42+d4t130a025s+d4t130a025s_s11+d4t130a025s_s23+t130s42".split("+")
D5 = "d5a040s+d5a040s_s11+d5a040s_s23+d5a040s_s42+d5a040s_s3+d5a040s_s5+d5a040s_s13+d5a040s_s17".split("+")
def rk(x):
    r = np.nan_to_num(O.rc_of(np.asarray(x, float), c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    return r
A, B = {}, {}
for wn, ix in W.items():
    A[wn] = rk(np.mean([np.load(OUT / ("h2_%s_%s.npy" % (t, wn))) for t in D4 + D5], axis=0))
    B[wn] = rk(build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP)
f = lambda v: "%.4f/%.3f/%.1f/%.3f/%.4f" % tuple(v[m] for m in MET)
for wn in W:
    ra, rb = G.ev(A[wn], wn), G.ev(B[wn], wn)
    line = "%s 1617 %s | 1610 %s" % (wn, f(ra), f(rb))
    for w in (0.2, 0.35, 0.5):
        r = G.ev((1 - w) * A[wn] + w * B[wn], wn)
        lin = {m: (1 - w) * ra[m] + w * rb[m] for m in MET}
        line += " | w%.2f %s（相对两端连线 IC_IR %+.3f 换手 %+.4f）" % (w, f(r), r["ic_ir"] - lin["ic_ir"], r["turnover"] - lin["turnover"])
    print(line, flush=True)
# (b) 分时段选方向
kd = pd.to_datetime(c.key_date)
SL = (1000, 1030, 1100, 1130, 1330, 1400, 1430)
pick = {}
for s0 in SL:
    sc = {"A": [], "B": []}
    for wn in ("2021", "2022", "2020"):
        for k, X in (("A", A[wn]), ("B", B[wn])):
            msk = np.asarray((kd.year == int(wn)) & (slot == s0) & np.isfinite(c.y))
            ics = c.per_date_rank_corr(X, np.nan_to_num(c.y), msk)
            ics = ics[np.isfinite(ics)]
            sc[k].append(ics.mean() / ics.std() if len(ics) > 2 else np.nan)
    pick[s0] = "A" if np.nanmean(sc["A"]) >= np.nanmean(sc["B"]) else "B"
    print("时段 %d：1617 IR %.3f，1610 IR %.3f → 选 %s" % (s0, np.nanmean(sc["A"]), np.nanmean(sc["B"]), "1617" if pick[s0] == "A" else "1610"), flush=True)
for wn in W:
    X = A[wn].copy()
    for s0, k in pick.items():
        if k == "B":
            m = slot == s0
            X[m] = B[wn][m]
    r, ra = G.ev(X, wn), G.ev(A[wn], wn)
    print("分时段选方向 %s %s（相对 1617 %+.5f/%+.4f/%+.2f/%+.4f/%+.4f）" % (wn, f(r), *[r[m] - ra[m] for m in MET]), flush=True)
