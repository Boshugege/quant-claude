# -*- coding: utf-8 -*-
"""主题"流动性冲击回吐"沿换手—IC 前沿取点，导出 2024 因子值送评分机估公榜分（同 c79 的做法）。
快信号：四个特质版本（特质段反转、特质放量／跳跃、特质主动买压、特质盘口订单流），截面秩平均后平滑 α；慢信号：PCUM、FLOW20（权重 ws）、OQ（权重 wq）。"""
import numpy as np
import pandas as pd
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, MET, OUT
from g_go import inst_codes
c = G.c; z = G.z930; did = c.date_id
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask=None):
    x = np.where((x != 0) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
mD = raw("D_R") != 0; PS = raw("P_SEG"); mP = mD & (PS != 0)
F4 = np.mean([-rk(raw("D_R") - PS, mP), -rk(raw("D_HV") + raw("D_JUMP") - PS, mP), -rk(raw("PD_MS_OFI")), -rk(raw("PD_CF_OFI3"))], axis=0)
PCUM, FLOW20 = rk(raw("P_CUM")), rk(raw("CF_OFI3_20"))
OQ = H.ewma_col(np.mean([rk(raw("OQ_NIMB")), -rk(raw("OQ_SIZE"))], axis=0), c.inst, 0.3)
SM = {a: H.ewma_col(F4, c.inst, a) for a in (0.6, 0.7, 0.8)}
CANDS = {"F1_a6_s25_q40": (0.6, 0.25, 0.40), "F2_a6_s40_q40": (0.6, 0.40, 0.40), "F3_a7_s25_q40": (0.7, 0.25, 0.40),
         "F4_a7_s40_q40": (0.7, 0.40, 0.40), "F5_a8_s25_q40": (0.8, 0.25, 0.40), "F6_a8_s40_q40": (0.8, 0.40, 0.40),
         "F7_a7_s60_q60": (0.7, 0.60, 0.60), "F8_a6_s60_q60": (0.6, 0.60, 0.60)}
kd = pd.to_datetime(c.key_date); m = (kd.year == 2024); ins = inst_codes(c)
for nm, (a, ws, wq) in CANDS.items():
    x = np.where(z, 0.0, SM[a] + ws * (PCUM + FLOW20) + wq * OQ)
    r = G.ev(x, "2024")
    print("%-16s 2024 %s" % (nm, "/".join("%.4f" % r[k] for k in MET)), flush=True)
    o = pd.DataFrame({"date": kd[m], "instrument": ins[np.flatnonzero(m)], "factor": x[m]}).drop_duplicates(["date", "instrument"], keep="last")
    o.to_parquet(OUT / ("val_%s_2024.parquet" % nm), index=False)
