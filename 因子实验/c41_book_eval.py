# -*- coding: utf-8 -*-
"""评估盘口深层新根（m4_book.py 的 9 个）：单独五窗口；方向按 2020–22（2021／2022／2020）的单独 IC 符号定；
再按锚的方式叠到 1610 与 1603（平台口径）：底座 + s·w·K·秩(根)，w = 0.025／0.05／0.075，五窗口五项（★ = 不差）。"""
import sys
import numpy as np
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10, K03 = 0.2774 / 0.289, 0.655 / 0.289
NAMES = sys.argv[1].split(",") if len(sys.argv) > 1 else [p + s for p in ("BK_THIN", "BK_CONS", "BK_DIMB") for s in ("", "_20", "_DEV")]
def rk(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
    x[x == 0] = np.nan
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    return r
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
BASES = {"1610": ({wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in W.items()}, K10),
         "1603": ({wn: np.where(z, 0.0, build("1603", ix, drop_deal=True)) for wn, ix in W.items()}, K03)}
REF = {b: {wn: G.ev(B[wn], wn) for wn in W} for b, (B, _) in BASES.items()}
for nm in NAMES:
    x = rk(nm)
    single = {wn: G.ev(x, wn) for wn in W}
    sg = 1.0 if np.mean([single[wn]["ic_mean"] for wn in ("2021", "2022", "2020")]) >= 0 else -1.0
    print("%-12s 单独 " % nm + " | ".join("%s %+.4f/%+.3f/%+.1f/%.3f/%.4f" % (wn, *[single[wn][m] for m in MET]) for wn in W) + "  → 方向 %+d" % sg, flush=True)
    for b, (B, K) in BASES.items():
        for w in (0.025, 0.05, 0.075):
            line, st = "   %s %+.3f·%s" % (b, sg * w, nm), 0
            for wn in W:
                r = G.ev(B[wn] + sg * w * K * x, wn)
                d = {m: r[m] - REF[b][wn][m] for m in MET}
                st += ok(d)
                line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
            print(line + "  %d/5" % st, flush=True)
