# -*- coding: utf-8 -*-
"""导出纯方向候选的因子值（2024 全年，date／instrument／factor），送 personal 评分机按值评分、对照参照池估公榜分。
定义同 c74（主题"流动性冲击回吐"），候选见 CANDS。"""
import numpy as np
import pandas as pd
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, OUT
from g_go import inst_codes
c = G.c; z = G.z930; did = c.date_id
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask=None):
    x = np.where((x != 0) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
mD = raw("D_R") != 0; PS = raw("P_SEG"); mP = mD & (PS != 0)
FAST = {"IDSEG": -rk(raw("D_R") - PS, mP), "IDHVJ": -rk(raw("D_HV") + raw("D_JUMP") - PS, mP),
        "TICK": -rk(raw("MS_OFI")), "COFI": -rk(raw("CF_OFI3"))}
PCUM = rk(raw("P_CUM"))
OQ = H.ewma_col(np.mean([rk(raw("OQ_NIMB")), -rk(raw("OQ_SIZE"))], axis=0), c.inst, 0.3)
FLOW20 = rk(raw("CF_OFI3_20"))
def comp(keys, a, slow=True):
    f = H.ewma_col(np.mean([FAST[k] for k in keys], axis=0), c.inst, a)
    return f + (0.25 * PCUM + 0.25 * OQ + 0.25 * FLOW20 if slow else 0.0)
CANDS = {"V1_a5_IDSEG_IDHVJ_slow": comp(["IDSEG", "IDHVJ"], 0.5), "V2_a5_four_slow": comp(["IDSEG", "IDHVJ", "TICK", "COFI"], 0.5),
         "V3_a6_IDSEG_IDHVJ_slow": comp(["IDSEG", "IDHVJ"], 0.6), "V4_a5_IDSEG_slow": comp(["IDSEG"], 0.5),
         "V5_a5_IDSEG_only": comp(["IDSEG"], 0.5, slow=False), "V6_a0_IDSEG_raw": -rk(raw("D_R") - PS, mP)}
kd = pd.to_datetime(c.key_date); m = (kd.year == 2024)
ins = inst_codes(c)
for nm, x in CANDS.items():
    x = np.where(z, 0.0, x)
    o = pd.DataFrame({"date": kd[m], "instrument": ins[np.flatnonzero(m)], "factor": x[m]}).drop_duplicates(["date", "instrument"], keep="last")
    o.to_parquet(OUT / ("val_%s_2024.parquet" % nm), index=False)
    print("导出", nm, len(o), flush=True)
