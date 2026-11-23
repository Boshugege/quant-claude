# -*- coding: utf-8 -*-
""""日频过热"组（10-19 夜）：52 周高点、波动率期限结构、成交量期限结构、资本利得悬置四个信号（都取反向）截面秩平均，
作为低换手锚叠到 1610、1603（平台口径）：底座 − w·K·组，w = 0.025／0.05／0.075，五窗口五项。"""
import numpy as np
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10, K03 = 0.2774 / 0.289, 0.655 / 0.289
def rk(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float); x[x == 0] = np.nan
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    return r
grp = np.mean([rk(n) for n in ("K_PH250", "K_VTS", "K_VOLTS", "K_CGO")], axis=0)
grp = np.nan_to_num(O.rc_of(grp, c.date_id, c.n_dates), nan=0.0); grp[z] = 0.0
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
print("过热组单独（取反向）" + " | ".join("%s %+.4f/%+.3f/%+.1f/%.3f/%.4f" % (wn, *[G.ev(-grp, wn)[m] for m in MET]) for wn in W), flush=True)
BASES = {"1610": ({wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in W.items()}, K10),
         "1603": ({wn: np.where(z, 0.0, build("1603", ix, drop_deal=True)) for wn, ix in W.items()}, K03)}
for b, (B, K) in BASES.items():
    ref = {wn: G.ev(B[wn], wn) for wn in W}
    for w in (0.025, 0.05, 0.075):
        line, st = "%s − %.3f·过热组" % (b, w), 0
        for wn in W:
            r = G.ev(B[wn] - w * K * grp, wn); d = {m: r[m] - ref[wn][m] for m in MET}
            st += ok(d)
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
        print(line + "  %d/5" % st, flush=True)
