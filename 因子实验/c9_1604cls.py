# -*- coding: utf-8 -*-
"""Claude 夜间 C9：1604（s152 + 六个安全根 + 三小根 + CLOSEPOS20）再加收盘段锚 CLS20，口径 engine.Core.evaluate。
CLS20 在 1398／1399 底座上 5/5 ★（§128–§129），但 152 底座上没测过带 14:30 加份的写法；C1 只测了 1602 底座的全天 0.03。
权重按秩口径（×底座截面标准差÷0.289），a = 全天，b = 14:30 再加。注意 CLS20 属同时段季节性一类，公榜 IC 可能打折（1232 只兑现 0.65）。
用法（personal）：~/jyc_scorer/.venv/bin/python 因子实验/c9_1604cls.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
from c1_upgrades import G, CP, CLS, ax, W1363, SMALL, DEAL, MET  # noqa: E402
from g_go import WINS, IXW  # noqa: E402

c = G.c
z930 = G.z930
slot = c.slot_of_date[c.date_id]
SAFE = [r for r in W1363 if r not in DEAL]
NAMES = ("fm_s152", "o21_s152", "o20_s152")


def base(ix):
    B = np.asarray(R.load(NAMES[ix])[0], float).copy()
    B[z930] = 0.0
    for r in SAFE:
        B += W1363[r] * ax(r, ix)
    for k, w in SMALL.items():
        B += w * G.axis(k, ix)
    sd = np.nanmedian([np.nanstd(B[(c.date_id == d)]) for d in np.unique(c.date_id)[::50]])
    B += 0.05 * sd / 0.289 * CP
    B[z930] = 0.0
    return B, sd


BS = {ix: base(ix) for ix in (0, 1, 2)}
BS[2] = (BS[2][0] + 0.05 * BS[0][1] / 0.289 * CP * (G.c.date_id >= 0), BS[0][1])   # o20 截面抽样落空（sd=0）→ 用主区间的 sd 补上 CP
print("截面标准差", {ix: round(BS[ix][1], 4) for ix in BS}, flush=True)


def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn, _, _ in WINS}


rb = ev5({ix: BS[ix][0] for ix in BS})
line = "1604 参考"
for wn in ("2023", "2024", "2021", "2022", "2020"):
    m = rb[wn]
    line += " | %s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"])
print(line, flush=True)
for a, b, e in ((0.05, 0.10, 0.0), (0.05, 0.10, 0.03), (0.05, 0.10, 0.05), (0.03, 0.06, 0.03), (0.05, 0.05, 0.03), (0.08, 0.16, 0.05)):
    Bs = {}
    for ix in (0, 1, 2):
        B, sd = BS[ix]
        k = sd / 0.289
        x = B + a * k * CLS + b * k * CLS * (slot == 1430) + e * k * CP
        x[z930] = 0.0
        Bs[ix] = x
    res = ev5(Bs)
    line = "CLS a%.2f b%.2f +CP%.2f" % (a, b, e)
    star = 0
    for wn in ("2023", "2024", "2021", "2022", "2020"):
        d = {m: res[wn][m] - rb[wn][m] for m in MET}
        ok = d["ic_mean"] >= 0 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
