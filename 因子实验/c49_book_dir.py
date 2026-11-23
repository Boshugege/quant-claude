# -*- coding: utf-8 -*-
"""新方向 D1（10-19）：盘口深层线性合成（只用今天新挖的盘口信息，与以往五个方向的信息源都不同）。
权重：2020–22 上逐截面岭回归（目标 = 标签截面秩，特征 = 各根截面秩，逐截面去均值），2023／2024 为样本外检验。
三个版本：快根、快根 + 慢根、快根 + 慢根 + CLS／CP 锚（降换手）。"""
import numpy as np
import pandas as pd
import d2_optimize as O
from g_go import GO
from c1_upgrades import OUT, CLS, CP
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
G = GO(); c = G.c; z = c.slot_of_date[c.date_id] == 930
def rk(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float); x[x == 0] = np.nan
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    return r
FAST = ["BK_DIMB", "BK_SLOPEI", "BK_CONS", "BK_RESIL", "BK_AOS_DEV", "BK_THIN"]
SLOW = ["BK_DIMB_20", "BK_SLOPEI_20", "BK_CONS_20", "BK_RESIL_20", "BK_AOS_20", "BK_THIN_20"]
F = {nm: rk(nm) for nm in FAST + SLOW}
F["CLS"], F["CP"] = CLS, CP
yr = pd.DatetimeIndex(c.day_of_date[c.date_id]).year
fit = np.flatnonzero((yr <= 2022) & ~z & np.isfinite(c.y))
g = c.date_id[fit]; nd = c.n_dates
ry = np.nan_to_num(O.rc_of(c.y[fit], g, nd), nan=0.0)
cnt = np.maximum(np.bincount(g, minlength=nd), 1).astype(float)[g]
def ridge(names, lam=1.0):
    X = np.column_stack([F[n][fit] for n in names])
    for j in range(X.shape[1]):
        X[:, j] -= np.bincount(g, X[:, j], nd)[g] / cnt
    yy = ry - np.bincount(g, ry, nd)[g] / cnt
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ yy)
for tag, names in (("快根", FAST), ("快 + 慢", FAST + SLOW), ("快 + 慢 + 锚", FAST + SLOW + ["CLS", "CP"])):
    w = ridge(names)
    w = w / np.abs(w).sum()
    B = sum(wi * F[n] for wi, n in zip(w, names))
    print("%s 权重 %s" % (tag, ", ".join("%s %+.3f" % (n, wi) for n, wi in zip(names, w))), flush=True)
    print("   " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[G.ev(B, wn)[m] for m in MET]) for wn in ("2023", "2024", "2021", "2022", "2020")), flush=True)
