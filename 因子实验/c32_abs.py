# -*- coding: utf-8 -*-
"""第九批四个文件与 1600 平台口径的本地五项绝对值（五个窗口），用于给用户的对照表。口径同 c27：去 Y_sd5、X_n5d，N_dir 活。"""
import json
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, W1363, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
def b1379(ix):
    B = np.asarray(G.B[("1363", ix)], float).copy()
    for r in W1363:
        B += (0.5 * W1363[r] - (1.5 * W1363[r] if r in U.DEAL else 0.0)) * G.axis(r, ix)
    B[z] = 0.0
    return B
K10, K11, K12 = 0.2774 / 0.289, 0.50 / 0.289, 0.544 / 0.289
D4 = ["d4t150a025s", "t150s11", "t150s23", "t150s42", "d4t130a025s", "d4t130a025s_s11", "d4t130a025s_s23", "t130s42"]
D5 = ["d5a040s", "d5a040s_s11", "d5a040s_s23", "d5a040s_s42", "d5a040s_s3", "d5a040s_s5", "d5a040s_s13", "d5a040s_s17"]
tree = lambda wn, tags: np.mean([np.load(OUT / ("h2_%s_%s.npy" % (t, wn))) for t in tags], axis=0)
F = {
    "1600平台": lambda wn, ix: tree(wn, ["seasplat"]),
    "1617": lambda wn, ix: tree(wn, D4 + D5),
    "1610": lambda wn, ix: build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP,
    "1611": lambda wn, ix: b1379(ix) + 0.05 * K11 * CLS + 0.10 * K11 * CLS * (slot == 1430) + 0.075 * K11 * CP,
    "1612": lambda wn, ix: build("1601", ix, drop_deal=True) + 0.01 * K12 * CLS + 0.02 * K12 * CLS * (slot == 1430) + 0.03 * K12 * CP,
}
res = {}
for name, f in F.items():
    res[name] = {}
    for wn, ix in W.items():
        r = G.ev(np.where(z, 0.0, f(wn, ix)), wn)
        res[name][wn] = {m: float(r[m]) for m in MET}
    print(name, json.dumps(res[name]), flush=True)
json.dump(res, open(OUT / "c32_abs.json", "w"), ensure_ascii=False, indent=1)
