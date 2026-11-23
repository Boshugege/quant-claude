# -*- coding: utf-8 -*-
"""1622（机制分组等风险合成）的半年稳健性：10 个半年里，相对 1610、1603（平台口径）各项不差的次数。"""
import numpy as np
import c1_upgrades as U
import c52_mech_group as M
from c1_upgrades import G, CLS, CP, build, MET
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
K10 = 0.2774 / 0.289
HY = [("%dH%d" % (y, h), "%d-%s" % (y, "01-01" if h == 1 else "07-01"), "%d-%s" % (y, "06-30" if h == 1 else "12-31")) for y in range(2020, 2025) for h in (1, 2)]
IX = lambda y: 0 if y >= 2023 else (1 if y >= 2021 else 2)
D = {ix: M.compose(ix, "c") for ix in (0, 1, 2)}
REF = {"1610": {ix: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for ix in (0, 1, 2)},
       "1603": {ix: np.where(z, 0.0, build("1603", ix, drop_deal=True)) for ix in (0, 1, 2)}}
for nm, R in REF.items():
    cnt = {m: 0 for m in MET}
    for h, st, en in HY:
        ix = IX(int(h[:4])); a = G.evspan(D[ix], st, en); b = G.evspan(R[ix], st, en)
        for m in MET:
            cnt[m] += (a[m] <= b[m]) if m == "turnover" else (a[m] >= b[m] - (0.00005 if m == "ic_mean" else 0.0))
    print("1622 相对 %s：10 个半年里不差的次数 IC %d IC_IR %d 多空 %d 压力 %d 换手 %d" % (nm, *[cnt[m] for m in MET]), flush=True)
