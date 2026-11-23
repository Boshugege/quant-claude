# -*- coding: utf-8 -*-
"""1622（机制分组等风险合成 D-c）的稳健性与变体（10-19 夜）：
(1) 锚点反转组权重 0.4／0.5／0.6（其余六组平分剩下的），看结果对这个经验选择是否敏感；
(2) 纯自研版：锚点反转组只留 490 头部（去掉队友 161），其余同 1622。"""
import numpy as np
import c52_mech_group as M
from c1_upgrades import G, MET
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
def comp(ix, aw, own=False):
    gs = M.groups(ix)
    gs.pop("盘口深层")
    if own:
        gs["锚点反转"] = gs["锚点反转"][1:]
    gz = {k: M.zs(np.mean(v, axis=0)) for k, v in gs.items()}
    w = {k: (aw if k == "锚点反转" else (1 - aw) / (len(gz) - 1)) for k in gz}
    return np.where(M.z, 0.0, sum(w[k] * gz[k] for k in gz))
for tag, aw, own in (("锚 0.4", 0.4, False), ("锚 0.5（1622）", 0.5, False), ("锚 0.6", 0.6, False), ("纯自研 锚 0.5", 0.5, True), ("纯自研 锚 0.6", 0.6, True)):
    print("%-14s " % tag + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[G.ev(comp(ix, aw, own), wn)[m] for m in MET]) for wn, ix in W.items()), flush=True)
