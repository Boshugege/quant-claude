# -*- coding: utf-8 -*-
"""Claude 夜间 C6：两族低换手组合的秩混合——152 族（1604：s152 + 六个安全根 + 三小根 + CLOSEPOS20）与 1363 族（1606 平台一致版）。
两族头部不同、相关不高，混合可能借分散化抬 IC_IR、换手落在两者之间。w = 1604 的权重。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
import root_eval as R  # noqa: E402
from c1_upgrades import G, CP, ax, W1363, SMALL, DEAL, MET, build  # noqa: E402
from g_go import WINS, IXW  # noqa: E402

c = G.c
z930 = G.z930
SAFE = [r for r in W1363 if r not in DEAL]
S152 = ("fm_s152", "o21_s152", "o20_s152")


def b1604(ix):
    B = np.asarray(R.load(S152[ix])[0], float).copy()
    B[z930] = 0.0
    B = np.nan_to_num(B, nan=0.0)
    for r in SAFE:
        B += W1363[r] * ax(r, ix)
    for k, w in SMALL.items():
        B += w * G.axis(k, ix)
    B += 0.047 * CP
    B[z930] = 0.0
    return B


def rk(B):
    r = np.nan_to_num(O.rc_of(B, c.date_id, c.n_dates), nan=0.0)
    r[z930] = 0.0
    return r


A = {ix: rk(b1604(ix)) for ix in (0, 1, 2)}
Bq = {ix: rk(build("1603", ix, dose=1.5, cp=0.05)) for ix in (0, 1, 2)}
for w in (0.0, 0.3, 0.5, 0.7, 1.0):
    Bs = {ix: w * A[ix] + (1 - w) * Bq[ix] for ix in (0, 1, 2)}
    line = "w1604=%.1f" % w
    for wn, _, _ in WINS:
        m = G.ev(Bs[IXW[wn]], wn)
        line += " | %s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"])
    print(line, flush=True)
mask = (z930 == False)  # noqa: E712
print("两族逐行秩相关（主区间）%.3f" % np.corrcoef(A[0][mask], Bq[0][mask])[0, 1], flush=True)
