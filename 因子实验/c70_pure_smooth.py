# -*- coding: utf-8 -*-
"""纯方向深挖 J3（10-20）：纯因子的平滑与快慢配对。纯因子的短板是换手（0.8 以上），这里看能压到多低、IC_IR 损失多少。
  平滑：按股票跨截面的指数平滑（h2_robust.ewma_col，α = 过去值的权重；1617 季节性那一列同法），α ∈ {0, 0.3, 0.5, 0.7, 0.85}
  配对：快信号 + w·慢版本（同一机制的 20 日常态），w ∈ {0.5, 1.0}
候选（截面秩，符号按单独方向）：OFI 主动买压回吐、IDSEG 特质段反转、PCUM 特质日内反转、OQ 委托结构（笔数失衡 + 规模比 ×2 + 慢版）、HVJ 放量与跳跃反转。
与参照池锚点（单机制因子，公榜 0.79～0.80）对照：2024 本地约 IC 0.068／IC_IR 1.17／多空 37／压力 1.10／换手 0.58。"""
import numpy as np
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, MET, OUT
c = G.c; z = G.z930; did = c.date_id
W = ("2023", "2024", "2021", "2022", "2020")
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask=None):
    x = np.where((x != 0) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
mD = raw("D_R") != 0
PS = raw("P_SEG")
C = {
    "OFI": (-rk(raw("MS_OFI")), -rk(raw("MS_OFI_20"))),
    "IDSEG": (-rk(raw("D_R") - PS, mD & (PS != 0)), rk(raw("P_5D"))),
    "PCUM": (rk(raw("P_CUM")), rk(raw("P_5D"))),
    "OQ": (np.mean([rk(raw("OQ_NIMB")), -rk(raw("OQ_SIZE")), -rk(raw("OQ_DSIZE"))], axis=0), rk(raw("OQ_NIMB_20"))),
    "HVJ": (-rk(raw("D_HV") + raw("D_JUMP"), mD), -rk(raw("MS_VR_20"))),
}
def show(tag, x):
    x = np.where(z, 0.0, x)
    r = {wn: G.ev(x, wn) for wn in W}
    print("%-30s " % tag + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
for nm, (fast, slow) in C.items():
    for a in (0.0, 0.3, 0.5, 0.7, 0.85):
        f = H.ewma_col(fast, c.inst, a) if a > 0 else fast
        show("%s α=%.2f" % (nm, a), f)
        if a in (0.0, 0.5, 0.7):
            for w in (0.5, 1.0):
                show("%s α=%.2f + %.1f·慢" % (nm, a, w), f + w * slow)
