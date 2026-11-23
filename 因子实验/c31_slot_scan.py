# -*- coding: utf-8 -*-
"""F4 弱时段：各根轴与 1610 在各截面时刻的 IC（2023+2024 主区间、2021+2022、2020），找 14:00 相对强的分量；
再试"只在 14:00 截面加份"（w~1400+x），五窗口五项，Δ 相对 1610。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, W1363, SMALL, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
W = ("2023", "2024", "2021", "2022", "2020")
K = 0.2774 / 0.289
B10 = {ix: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K * CLS + 0.10 * K * CLS * (slot == 1430) + 0.075 * K * CP) for ix in (0, 1, 2)}
def slots(B, wn):
    _, d = c.evaluate(np.where(z, 0.0, B), start=[w for w in WINS if w[0] == wn][0][1], end=[w for w in WINS if w[0] == wn][0][2] + " 23:59:59")
    return d
d0 = slots(B10[0], "2023")
print("evaluate 明细的键：", list(d0.keys())[:12], flush=True)
SL = [1000, 1030, 1100, 1130, 1330, 1400, 1430]
def per_slot(B, wn):
    d = slots(B, wn)
    return {s: float(np.nanmean(d["ic"][d["slot"] == s])) for s in SL}
def agg(Bs):
    out = {}
    for wn in W:
        out[wn] = per_slot(Bs[IXW[wn]], wn)
    return {s: np.mean([out[wn][s] for wn in W]) for s in SL}
p10 = agg(B10)
print("1610 各时段 IC（五窗口平均）：" + " ".join("%d:%.4f" % (s, p10[s]) for s in SL), flush=True)
LABS = [r for r in W1363 if r not in U.DEAL] + list(SMALL) + ["s155", "s139", "s161"]
cand = []
for lab in LABS + ["CLS", "CP"]:
    Ax = {ix: (CLS if lab == "CLS" else CP if lab == "CP" else G.axis(lab, ix)) for ix in (0, 1, 2)}
    p = agg(Ax)
    m = np.mean([p[s] for s in SL if s != 1400])
    print("%-8s 14:00 %.4f  其余均值 %.4f  比 %.2f  | " % (lab, p[1400], m, p[1400] / m if m else 0) + " ".join("%d:%.4f" % (s, p[s]) for s in SL), flush=True)
    cand.append((abs(p[1400]) - abs(m), lab, np.sign(p[1400]), Ax))
cand.sort(key=lambda t: -t[0])
REF = {wn: {m: G.ev(B10[IXW[wn]], wn)[m] for m in MET} for wn in W}
for _, lab, sg, Ax in cand[:4]:
    for w in (0.03, 0.06):
        Bs = {ix: np.where(z, 0.0, B10[ix] + sg * w * K * Ax[ix] * (slot == 1400)) for ix in (0, 1, 2)}
        line, star = "1610 + %s 只在 14:00 ×%+.2f" % (lab, sg * w), 0
        for wn in W:
            r = G.ev(Bs[IXW[wn]], wn); d = {m: r[m] - REF[wn][m] for m in MET}
            ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            star += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % star, flush=True)
