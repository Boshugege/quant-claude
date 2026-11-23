# -*- coding: utf-8 -*-
"""10-20：快信号（段内主动买压 MS_OFI，反向）抬 IC、换手升；慢锚压换手、伤多空。两者成对叠到 1610／1603，看能否五项都不差。
慢锚两组（组内截面秩等权，符号按经济先验，与第二、三批单独结果一致）：
  微观慢组  −MS_AC1_20（价差弹跳少）、+MS_JUMP_20（信息冲击多）、−MS_VR_20（分钟级趋势少）
  投机热度组 −K_MAX20、−K_PVC20、−K_SKEW20、−K_REV20
底座 + a·K·秩(−OFI) + b·K·秩(慢组)，a ∈ {0.025, 0.05}，b ∈ {0.025, 0.05, 0.075}；五窗口五项（★ = 不差）。"""
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
def grp(items):
    g = np.mean([s * rk(nm) for s, nm in items], axis=0)
    r = np.nan_to_num(O.rc_of(np.where(g == 0, np.nan, g), c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
    return r
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
BASES = {"1610": ({wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in W.items()}, K10),
         "1603": ({wn: np.where(z, 0.0, build("1603", ix, drop_deal=True)) for wn, ix in W.items()}, K03)}
REF = {b: {wn: G.ev(B[wn], wn) for wn in W} for b, (B, _) in BASES.items()}
OFI = -rk("MS_OFI")
SLOW = {"微观慢组": grp([(-1, "MS_AC1_20"), (1, "MS_JUMP_20"), (-1, "MS_VR_20")]),
        "投机热度组": grp([(-1, "K_MAX20"), (-1, "K_PVC20"), (-1, "K_SKEW20"), (-1, "K_REV20")])}
SLOW["两组合并"] = 0.5 * (SLOW["微观慢组"] + SLOW["投机热度组"])
for nm, x in SLOW.items():
    single = {wn: G.ev(x, wn) for wn in W}
    print("%-8s 单独 " % nm + " | ".join("%s %+.4f/%+.3f/%+.1f/%.3f/%.4f" % (wn, *[single[wn][m] for m in MET]) for wn in W), flush=True)
for b, (B, K) in BASES.items():
    for nm, x in SLOW.items():
        for a in (0.025, 0.05):
            for w in (0.025, 0.05, 0.075):
                line, st = "   %s %+.3f·OFI %+.3f·%s" % (b, a, w, nm), 0
                for wn in W:
                    r = G.ev(B[wn] + a * K * OFI + w * K * x, wn)
                    d = {m: r[m] - REF[b][wn][m] for m in MET}
                    st += ok(d)
                    line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
                print(line + "  %d/5" % st, flush=True)
