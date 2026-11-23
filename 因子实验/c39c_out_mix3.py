# -*- coding: utf-8 -*-
"""E7c（10-19）：三者混合 y = (1−a−b)·z(原始) + a·z(秩) + b·z(正态分位数)；同一组 (a, b) 要在四个方向 × 五窗口都不差才算通用改进。在 E7 的基础上：平台评分前对因子做风格回归取残差，这一步与取值分布有关，所以截面秩、正态分位数、缩尾可能改变五项。
对 1617、1610、1612、1603（平台口径）各试 rank（均匀）、gauss（正态分位数）、win1（截面 1%／99% 缩尾），相对原始输出比五窗口五项。"""
import numpy as np
import pandas as pd
from scipy.special import erfinv
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10, K12 = 0.2774 / 0.289, 0.544 / 0.289
D4 = "d4t150a025s+t150s11+t150s23+t150s42+d4t130a025s+d4t130a025s_s11+d4t130a025s_s23+t130s42".split("+")
D5 = "d5a040s+d5a040s_s11+d5a040s_s23+d5a040s_s42+d5a040s_s3+d5a040s_s5+d5a040s_s13+d5a040s_s17".split("+")
F = {
    "1617": lambda wn, ix: np.mean([np.load(OUT / ("h2_%s_%s.npy" % (t, wn))) for t in D4 + D5], axis=0),
    "1610": lambda wn, ix: build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP,
    "1612": lambda wn, ix: build("1601", ix, drop_deal=True) + 0.01 * K12 * CLS + 0.02 * K12 * CLS * (slot == 1430) + 0.03 * K12 * CP,
    "1603": lambda wn, ix: build("1603", ix, drop_deal=True),
}
did = c.date_id
def rank(x):
    return np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0)
def gauss(x):
    r = rank(x)
    return np.sqrt(2.0) * erfinv(np.clip(2.0 * r, -0.999, 0.999))
def win1(x):
    s = pd.Series(x)
    g = s.groupby(did)
    lo, hi = g.transform(lambda v: v.quantile(0.01)), g.transform(lambda v: v.quantile(0.99))
    return s.clip(lo, hi).to_numpy()
def zs(x):
    s = pd.Series(x); g = s.groupby(did)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()
def mix(a):
    return lambda x: (1 - a) * zs(x) + a * zs(rank(x))
def winq(q):
    def f(x):
        s = pd.Series(x); g = s.groupby(did)
        return s.clip(g.transform(lambda v: v.quantile(q)), g.transform(lambda v: v.quantile(1 - q))).to_numpy()
    return f
def mix3(a, b):
    return lambda x: (1 - a - b) * zs(x) + a * zs(rank(x)) + b * zs(gauss(x))
T = {"a%.2fb%.2f" % (a, b): mix3(a, b) for a in (0.15, 0.25) for b in (0.10, 0.20, 0.30)}
for nm, f in F.items():
    for wn, ix in W.items():
        x = np.where(z, 0.0, f(wn, ix)).astype(float)
        ref = G.ev(x, wn)
        line = "%s %s 原始 %.4f/%.3f/%.1f/%.3f/%.4f" % (nm, wn, *[ref[m] for m in MET])
        for tn, tf in T.items():
            y = np.where(z, 0.0, tf(x))
            r = G.ev(y, wn)
            d = {m: r[m] - ref[m] for m in MET}
            ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (tn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line, flush=True)
