# -*- coding: utf-8 -*-
"""第九批的半年稳健性：10 个半年（2020H1…2024H2），每个文件相对其公榜版本（平台口径）各项变好的半年数。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, W1363, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
HY = [("%dH1" % y, "%d-01-01" % y, "%d-06-30" % y) for y in range(2020, 2025)] + [("%dH2" % y, "%d-07-01" % y, "%d-12-31" % y) for y in range(2020, 2025)]
IX = lambda y: 0 if y >= 2023 else (1 if y >= 2021 else 2)
def ev(B, st, en):
    return G.evspan(B, st, en)
def b1379(ix):
    B = np.asarray(G.B[("1363", ix)], float).copy()
    for r in W1363:
        B += (0.5 * W1363[r] - (1.5 * W1363[r] if r in U.DEAL else 0.0)) * G.axis(r, ix)
    B[z] = 0.0
    return B
K10, K11, K12 = 0.2774 / 0.289, 0.50 / 0.289, 0.544 / 0.289
pairs = {
    "1610 vs 1602": (lambda ix: build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP, lambda ix: build("1602", ix, drop_deal=True)),
    "1611 vs 1379": (lambda ix: b1379(ix) + 0.05 * K11 * CLS + 0.10 * K11 * CLS * (slot == 1430) + 0.075 * K11 * CP, b1379),
    "1612 vs 1601": (lambda ix: build("1601", ix, drop_deal=True) + 0.01 * K12 * CLS + 0.02 * K12 * CLS * (slot == 1430) + 0.03 * K12 * CP, lambda ix: build("1601", ix, drop_deal=True)),
}
TAGS8 = ["d4t150a025s", "t150s11", "t150s23", "t150s42", "d4t130a025s", "d4t130a025s_s11", "d4t130a025s_s23", "t130s42"]
WY = {2023: "2023", 2024: "2024", 2021: "2021", 2022: "2022", 2020: "2020"}
def tree(y, tags):
    return np.where(z, 0.0, np.mean([np.load(OUT / ("h2_%s_%s.npy" % (t, WY[y]))) for t in tags], axis=0))
for name, (fa, fb) in pairs.items():
    A = {ix: fa(ix) for ix in (0, 1, 2)}; Bb = {ix: fb(ix) for ix in (0, 1, 2)}
    cnt = {m: 0 for m in MET}
    for nm, st, en in HY:
        y = int(nm[:4]); ra = ev(np.where(z, 0.0, A[IX(y)]), st, en); rb = ev(np.where(z, 0.0, Bb[IX(y)]), st, en)
        for m in MET:
            cnt[m] += (ra[m] <= rb[m]) if m == "turnover" else (ra[m] >= rb[m] - (0.00005 if m == "ic_mean" else 0.0))
    print("%-14s 半年里不差的次数（共 10）：IC %d  IC_IR %d  多空 %d  压力 %d  换手 %d" % (name, *[cnt[m] for m in MET]), flush=True)
for name, tags in (("1615(八树) vs 1600平台", TAGS8), ("1614(两树) vs 1600平台", ["d4t150a025s", "d4t130a025s_s11"])):
    cnt = {m: 0 for m in MET}
    for nm, st, en in HY:
        y = int(nm[:4])
        ra = ev(tree(y, tags), st, en); rb = ev(tree(y, ["seasplat"]), st, en)
        for m in MET:
            cnt[m] += (ra[m] <= rb[m]) if m == "turnover" else (ra[m] >= rb[m] - (0.00005 if m == "ic_mean" else 0.0))
    print("%-14s 半年里不差的次数（共 10）：IC %d  IC_IR %d  多空 %d  压力 %d  换手 %d" % (name, *[cnt[m] for m in MET]), flush=True)
