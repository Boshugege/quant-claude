# -*- coding: utf-8 -*-
"""稀疏可解释组合的五窗口检查（10-20 夜，macstudio CPU）：1610 + 少数几项（与 c95 b5 同一组候选），
替代 c92 的 12 项定权。每项逐截面意义清楚：
  Q   委托结构：深档（2–3 档）笔数失衡 + 一档平均委托规模比（取负），平滑 0.3
  IQ  同上的特质版（减去同伴均值）
  SP  价差常态（20 日）  FL 订单流常态（20 日）
  TK  特质主动买压的延续（PD_MS_OFI 正向，平滑 0.6；c92 里在其它项给定时为负权重 = 延续）
  HV  特质放量／跳跃段反转（平滑 0.6）  VW 特质 VWAP 偏离（平滑 0.3）
合成 = 1610/σ + Σ w·项/σ；五窗口（2023、2024 用 2020–22 拟合的 1610；2021、2022、2020 用各自窗口）对 1610 比五项，★ = 五项不差。"""
import numpy as np
import c1_upgrades as U
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; did = c.date_id; slot = c.slot_of_date[did]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)                         # noqa: E731
def rk(x, mask=None):
    x = np.where((np.isfinite(x) & (x != 0)) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
sm = lambda x, a: H.ewma_col(x, c.inst, a)                                               # noqa: E731
zz = lambda v: np.where(z, 0.0, v / (v[~z].std() + 1e-12))                              # noqa: E731
DR, PS = raw("D_R"), raw("P_SEG"); mP = (DR != 0) & (PS != 0)
P = {"Q": zz(sm(np.mean([rk(raw("QL_NIMB23")), -rk(raw("OQ_SIZE"))], axis=0), 0.3)),
     "IQ": zz(sm(np.mean([rk(raw("PD_QL_NIMB23")), -rk(raw("PD_OQ_SIZE"))], axis=0), 0.3)),
     "N23": zz(sm(rk(raw("QL_NIMB23")), 0.3)), "S1": zz(sm(-rk(raw("OQ_SIZE")), 0.3)),
     "SP": zz(rk(raw("AN_SPRD_20"))), "FL": zz(rk(raw("CF_OFI3_20"))),
     "TK": zz(sm(rk(raw("PD_MS_OFI")), 0.6)), "HV": zz(sm(-rk(raw("D_HV") + raw("D_JUMP") - PS, mP), 0.6)),
     "VW": zz(sm(-rk(raw("PD_AN_VWAP")), 0.3))}
B = {ix: np.nan_to_num(np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP)) for ix in (0, 1, 2)}
S = {ix: B[ix][~z].std() for ix in B}
REF = {wn: G.ev(B[ix], wn) for wn, ix in W.items()}
print("1610 " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[REF[wn][m] for m in MET]) for wn in W), flush=True)
C = {"q20": dict(Q=0.2), "q30": dict(Q=0.3), "q40": dict(Q=0.4), "iq30": dict(IQ=0.3), "n23_30": dict(N23=0.3), "s1_30": dict(S1=0.3),
     "q30_sp15": dict(Q=0.3, SP=0.15), "q30_sp15_fl10": dict(Q=0.3, SP=0.15, FL=0.1), "q30_sp15_tk15": dict(Q=0.3, SP=0.15, TK=0.15),
     "q30_sp15_hv15": dict(Q=0.3, SP=0.15, HV=0.15), "q30_sp15_tk15_hv15": dict(Q=0.3, SP=0.15, TK=0.15, HV=0.15),
     "q30_sp15_tk15_hv15_vw10": dict(Q=0.3, SP=0.15, TK=0.15, HV=0.15, VW=0.1), "q40_sp20_tk20_hv15": dict(Q=0.4, SP=0.2, TK=0.2, HV=0.15),
     "q30_tk15": dict(Q=0.3, TK=0.15)}
for name, w in C.items():
    line, st = "  %-26s" % name, 0
    for wn, ix in W.items():
        x = np.where(z, 0.0, B[ix] / S[ix] + sum(v * P[k] for k, v in w.items()))
        r = G.ev(x, wn); d = {m: r[m] - REF[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        st += ok
        line += " | %s %+.4f/%+.3f/%+.2f/%+.3f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % st, flush=True)
