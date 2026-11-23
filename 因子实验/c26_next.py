# -*- coding: utf-8 -*-
"""明天的升级候选（平台口径：去 Y_sd5、X_n5d，N_dir 活）：
 (1) 1610 + w·rank(s152)（两个队友头部）；(2) 1611 + 三小根（1379 原本没有 k1381／k1385／k1406）。Δ 相对 1610／1611。"""
import numpy as np
import c1_upgrades as U
import root_eval as R
from c1_upgrades import G, CP, CLS, build, W1363, SMALL, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = ("2023", "2024", "2021", "2022", "2020")
U.DEAL = {"Y_sd5", "X_n5d"}
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn in W}
def rep(name, ref, Bs):
    r = ev5(Bs); line, star = name, 0
    for wn in W:
        d = {m: r[wn][m] - ref[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
b1610 = {ix: build("1602", ix, drop_deal=True) + 0.048 * CLS + 0.096 * CLS * (slot == 1430) + 0.072 * CP for ix in (0, 1, 2)}
r1610 = ev5(b1610)
S152 = {}
for ix, nm in enumerate(("fm_s152", "o21_s152", "o20_s152")):
    x = np.asarray(R.load(nm)[0], float); S152[ix] = np.nan_to_num(U.O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
for w in (0.1, 0.2, 0.3):
    rep("1610 + s152×%.1f" % w, r1610, {ix: np.where(z, 0.0, b1610[ix] + w * S152[ix]) for ix in (0, 1, 2)})
def b1611(ix):
    B = np.asarray(G.B[("1363", ix)], float).copy()
    for r in W1363:
        B += (0.5 * W1363[r] - (1.5 * W1363[r] if r in ("Y_sd5", "X_n5d") else 0.0)) * G.axis(r, ix)
    B = B + 0.0865 * CLS + 0.173 * CLS * (slot == 1430) + 0.1298 * CP
    B[z] = 0.0
    return B
B11 = {ix: b1611(ix) for ix in (0, 1, 2)}
r1611 = ev5(B11)
for f in (0.5, 1.0, 1.5):
    rep("1611 + 三小根×%.1f" % f, r1611, {ix: np.where(z, 0.0, B11[ix] + sum(f * w * G.axis(k, ix) for k, w in SMALL.items())) for ix in (0, 1, 2)})
