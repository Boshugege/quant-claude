# -*- coding: utf-8 -*-
"""10-20：盘口委托笔数（m15）叠到 1610／1603 的细化。单项与组合，权重更细，另配慢版本 OQ_NIMB_20 压换手。
项（截面秩，符号按单独五窗口一致的方向）：S = −OQ_SIZE，D = −OQ_DSIZE，N = +OQ_NIMB，N20 = +OQ_NIMB_20。
五窗口五项（★ = 不差），另报 2023／2024 两年都 ★ 的组合。"""
import numpy as np
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10, K03 = 0.2774 / 0.289, 0.655 / 0.289
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
T = {"S": -rk("OQ_SIZE"), "D": -rk("OQ_DSIZE"), "N": rk("OQ_NIMB"), "N20": rk("OQ_NIMB_20")}
COMBOS = []
for t in ("S", "D", "N"):
    for w in (0.0125, 0.025, 0.0375, 0.05):
        COMBOS.append({t: w})
for w in (0.0125, 0.025, 0.0375):
    COMBOS.append({"S": w, "N": w})
    COMBOS.append({"D": w, "N": w})
    for w2 in (0.025, 0.05):
        COMBOS.append({"S": w, "N20": w2})
        COMBOS.append({"D": w, "N20": w2})
        COMBOS.append({"S": w, "N": w, "N20": w2})
for b, (B, K) in BASES.items():
    for cb in COMBOS:
        line, st, oos = "   %s + %s" % (b, " + ".join("%.4f·%s" % (w, t) for t, w in cb.items())), 0, 0
        for wn in W:
            r = G.ev(B[wn] + K * sum(w * T[t] for t, w in cb.items()), wn)
            d = {m: r[m] - REF[b][wn][m] for m in MET}
            st += ok(d); oos += ok(d) and wn in ("2023", "2024")
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
        print(line + "  %d/5%s" % (st, "  样本外 2/2" if oos == 2 else ""), flush=True)
