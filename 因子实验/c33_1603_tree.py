# -*- coding: utf-8 -*-
"""1603 方向的一次预先定好的检验：把深度 6 八种子平均树按秩权重 w ∈ {0.1, 0.2, 0.3} 混进 1603（平台口径），对照 1603 本身，五窗口五项。
两者都先做截面秩（−0.5～0.5），再按 (1−w)·1603 + w·树 合成。"""
import numpy as np
import pandas as pd
import c1_upgrades as U
from c1_upgrades import G, build, MET, OUT
c = G.c; z = G.z930
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
D6 = ["d6t85a050s"] + ["d6t85a050s_s%d" % s for s in (11, 23, 42, 3, 5, 13, 17)]
did = c.date_id
def rk(x):
    return pd.Series(x).groupby(did).rank(pct=True).to_numpy() - 0.5
for wn, ix in W.items():
    base = np.where(z, 0.0, build("1603", ix, drop_deal=True))
    tr = np.where(z, 0.0, np.mean([np.load(OUT / ("h2_%s_%s.npy" % (t, wn))) for t in D6], axis=0))
    rb, rt = rk(base), rk(tr)
    ref = G.ev(base, wn)
    line = "%s 1603=%.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[m] for m in MET])
    for w in (0.1, 0.2, 0.3):
        r = G.ev(np.where(z, 0.0, (1 - w) * rb + w * rt), wn)
        d = {m: r[m] - ref[m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        line += " | w%.1f %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (w, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line, flush=True)
