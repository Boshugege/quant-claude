# -*- coding: utf-8 -*-
"""第九批树方向的半年稳健性补算：1617（深度 4 ×8 + 深度 5 ×8）与 1616（深度 5 ×4），相对 1600 平台口径（seasplat），10 个半年各项不差的次数。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, MET, OUT
z = G.z930
U.DEAL = {"Y_sd5", "X_n5d"}
HY = [("%dH%d" % (y, h), "%d-%s" % (y, "01-01" if h == 1 else "07-01"), "%d-%s" % (y, "06-30" if h == 1 else "12-31")) for y in range(2020, 2025) for h in (1, 2)]
D4 = ["d4t150a025s", "t150s11", "t150s23", "t150s42", "d4t130a025s", "d4t130a025s_s11", "d4t130a025s_s23", "t130s42"]
D5A = ["d5a040s", "d5a040s_s11", "d5a040s_s23", "d5a040s_s42"]
D5B = ["d5a040s_s3", "d5a040s_s5", "d5a040s_s13", "d5a040s_s17"]
def tree(y, tags):
    return np.where(z, 0.0, np.mean([np.load(OUT / ("h2_%s_%d.npy" % (t, y))) for t in tags], axis=0))
for name, tags in (("1617(16树) vs 1600平台", D4 + D5A + D5B), ("1616(深5四树) vs 1600平台", D5A), ("1615(八树) vs 1600平台", D4)):
    cnt = {m: 0 for m in MET}
    for nm, st, en in HY:
        y = int(nm[:4])
        ra = G.evspan(tree(y, tags), st, en); rb = G.evspan(tree(y, ["seasplat"]), st, en)
        for m in MET:
            cnt[m] += (ra[m] <= rb[m]) if m == "turnover" else (ra[m] >= rb[m] - (0.00005 if m == "ic_mean" else 0.0))
    print("%-22s 半年里不差的次数（共 10）：IC %d  IC_IR %d  多空 %d  压力 %d  换手 %d" % (name, *[cnt[m] for m in MET]), flush=True)
