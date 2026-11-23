# -*- coding: utf-8 -*-
"""F1：1610 = 1602（公榜 0.86667：队友 161 头部 + 九根 + 三小根）+ CLS20／CP 锚对，五窗口五项，Δ 相对 1602。
两个底座口径都要过：全含（= 已交的 1602）与去 Y_sd5+X_n5d（公榜换手更像这个，N_dir 活）。
权重为秩口径 × k，k = 1602 截面标准差 0.2774 ÷ 0.289（c1_upgrades 的 sd）。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
K = 0.2774 / 0.289
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn, _, _ in WINS}
for nm, dset in (("全含", set()), ("去Y+X", {"Y_sd5", "X_n5d"})):
    U.DEAL = dset
    base = {ix: build("1602", ix, drop_deal=True) for ix in (0, 1, 2)}
    ref = ev5(base)
    print("== 底座 %s：" % nm + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in ("2023", "2024", "2021", "2022", "2020")), flush=True)
    for a, b, e in ((0.05, 0.10, 0.05), (0.05, 0.10, 0.075), (0.05, 0.10, 0.10), (0.03, 0.06, 0.05), (0.05, 0.05, 0.05), (0.0, 0.0, 0.05)):
        Bs = {}
        for ix in (0, 1, 2):
            x = base[ix] + a * K * CLS + b * K * CLS * (slot == 1430) + e * K * CP
            x[z] = 0.0
            Bs[ix] = x
        r = ev5(Bs)
        line, star = "%s CLS%.2f+%.2f CP%.3f" % (nm, a, b, e), 0
        for wn in ("2023", "2024", "2021", "2022", "2020"):
            d = {m: r[wn][m] - ref[wn][m] for m in MET}
            ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            star += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % star, flush=True)
