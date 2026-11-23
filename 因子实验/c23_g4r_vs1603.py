# -*- coding: utf-8 -*-
"""1603 方向的平台口径重新定权（g_opt --sub Y_sd5,X_n5d，λ = 1／2／3）加上 1603 原有的 CLS20（0.115，14:30 +0.23），
与 1603 平台口径（c1_upgrades.build("1603")，去 Y+X）比五个窗口五项。也试再加 CP 锚。"""
import json, sys
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, W1363, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = ("2023", "2024", "2021", "2022", "2020")
U.DEAL = {"Y_sd5", "X_n5d"}
runs = json.load(open(sys.argv[1], encoding="utf-8"))["runs"]
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn in W}
ref = ev5({ix: build("1603", ix, drop_deal=True) for ix in (0, 1, 2)})
print("1603平台 " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
def cand(ix, delta, cp):
    B = np.asarray(G.B[("1363", ix)], float).copy() - sum(W1363[r] * G.axis(r, ix) for r in ("Y_sd5", "X_n5d"))
    for l, v in delta.items():
        if abs(v) > 1e-12:
            B = B + v * G.axis(l, ix)
    B = B + 0.115 * CLS + 0.23 * CLS * (slot == 1430) + cp * 0.655 / 0.289 * CP
    B[z] = 0.0
    return B
for k, r in sorted(runs.items()):
    for cp in (0.0, 0.03):
        res = ev5({ix: cand(ix, r["delta"], cp) for ix in (0, 1, 2)})
        line, star = "λ%s CP%.2f" % (k, cp), 0
        for wn in W:
            d = {m: res[wn][m] - ref[wn][m] for m in MET}
            ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            star += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % star, flush=True)
