# -*- coding: utf-8 -*-
"""1622 的锚点反转组扩展（10-19 夜）：同一机制（价格锚点反转、同伴组内比较）的多个估计取平均以降噪。
A0 = 1622（490 + s161）；A1 = 490 + s161 + s152；A2 = 490 + s161 + s152 + s155；A3 = 490 + s161 + s152 + s155 + s139。其余同 1622。"""
import numpy as np
import h2_robust as H
import c52_mech_group as M
from c1_upgrades import G, MET
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
def comp(ix, extra):
    gs = M.groups(ix)
    gs.pop("盘口深层")
    for nm in extra:
        gs["锚点反转"].append(M.rk(H.s152_axis(G, ix)) if nm == "s152" else M.rk(G.axis(nm, ix)))
    gz = {k: M.zs(np.mean(v, axis=0)) for k, v in gs.items()}
    w = {k: (0.5 if k == "锚点反转" else 0.5 / (len(gz) - 1)) for k in gz}
    return np.where(M.z, 0.0, sum(w[k] * gz[k] for k in gz))
for tag, extra in (("A0 1622", []), ("A1 +152", ["s152"]), ("A2 +152+155", ["s152", "s155"]), ("A3 +152+155+139", ["s152", "s155", "s139"])):
    print("%-16s " % tag + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[G.ev(comp(ix, extra), wn)[m] for m in MET]) for wn, ix in W.items()), flush=True)
