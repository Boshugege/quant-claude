# -*- coding: utf-8 -*-
"""1603 的合规升级候选（10-19 下午）：平台口径 + 0.5×(Y_sd5／X_n5d 复活后的差) − r·K·秩(BK_RESIL) − k·K·秩(BK_CONS)。
4 个预先定好的组合：(r, k) ∈ {0, 0.0125} × {0, 0.0125}。五窗口五项，另打印 10 个半年的"不差"次数。"""
import itertools
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, build, MET, OUT
c = G.c; z = G.z930
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K03 = 0.655 / 0.289
HY = [("%dH%d" % (y, h), "%d-%s" % (y, "01-01" if h == 1 else "07-01"), "%d-%s" % (y, "06-30" if h == 1 else "12-31")) for y in range(2020, 2025) for h in (1, 2)]
IX = lambda y: 0 if y >= 2023 else (1 if y >= 2021 else 2)
def rk(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float); x[x == 0] = np.nan
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    return r
RES, CONS = rk("BK_RESIL"), rk("BK_CONS")
def b(ix, full):
    U.DEAL = set() if full else {"Y_sd5", "X_n5d"}
    return np.where(z, 0.0, build("1603", ix, drop_deal=not full))
P = {ix: b(ix, False) for ix in (0, 1, 2)}
D = {ix: b(ix, True) - P[ix] for ix in (0, 1, 2)}
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
ref = {wn: G.ev(P[ix], wn) for wn, ix in W.items()}
for r_, k_ in itertools.product((0.0, 0.0125), (0.0, 0.0125)):
    C = {ix: P[ix] + 0.5 * D[ix] - K03 * (r_ * RES + k_ * CONS) for ix in (0, 1, 2)}
    line, st = "1603 +0.5复活 RESIL-%.4f CONS-%.4f" % (r_, k_), 0
    for wn, ix in W.items():
        q = G.ev(C[ix], wn); d = {m: q[m] - ref[wn][m] for m in MET}
        st += ok(d)
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
    cnt = {m: 0 for m in MET}
    for nm, st_, en_ in HY:
        ix = IX(int(nm[:4])); qa = G.evspan(C[ix], st_, en_); qb = G.evspan(P[ix], st_, en_)
        for m in MET:
            cnt[m] += (qa[m] <= qb[m]) if m == "turnover" else (qa[m] >= qb[m] - (0.00005 if m == "ic_mean" else 0.0))
    print(line + "  %d/5 | 半年 %s" % (st, "/".join(str(cnt[m]) for m in MET)), flush=True)
