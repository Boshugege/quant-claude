# -*- coding: utf-8 -*-
"""纯方向（10-20，用户："纯纯的在树上堆叠没意思，去发掘别的因子单纯的方向"）：每个机制自成一个因子，只在机制内部组合（快慢配对、扣同伴），
不与别的机制混合、不叠加。五窗口五项（平台口径），用来挑做代码版的 2～3 个。
  P1 主动买压回吐      −秩(MS_OFI)；+ 慢版本（−MS_OFI_20）配对
  P2 特质段反转        −秩(D_R − P_SEG)
  P3 特质日内反转      +秩(P_CUM)；+ 特质多日（P_5D）配对
  P4 委托结构          秩(OQ_NIMB) − 秩(OQ_SIZE) − 秩(OQ_DSIZE)；+ 慢版本 OQ_NIMB_20
  P5 放量反转          −秩(D_HV)；放量 + 跳跃（都去掉缩量漂移）
  P6 特质放量反转      −秩(D_HV + D_JUMP − P_SEG)（只反转"放量或跳跃、且不随同伴"的那部分）
组内各项截面秩等权（符号按单独五窗口一致的方向），组合后不再加权。"""
import numpy as np
import d2_optimize as O
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
F = {
    "OFI": -rk(raw("MS_OFI")), "OFI20": -rk(raw("MS_OFI_20")),
    "IDSEG": -rk(raw("D_R") - PS, mD & (PS != 0)),
    "PCUM": rk(raw("P_CUM")), "P5D": rk(raw("P_5D")),
    "NIMB": rk(raw("OQ_NIMB")), "SIZE": -rk(raw("OQ_SIZE")), "DSIZE": -rk(raw("OQ_DSIZE")), "NIMB20": rk(raw("OQ_NIMB_20")),
    "HV": -rk(raw("D_HV"), mD), "HVJ": -rk(raw("D_HV") + raw("D_JUMP"), mD),
    "IDHVJ": -rk(raw("D_HV") + raw("D_JUMP") - PS, mD & (PS != 0)),
}
DIRS = {
    "P1 主动买压回吐": [["OFI"], ["OFI", "OFI20"]],
    "P2 特质段反转": [["IDSEG"]],
    "P3 特质日内反转": [["PCUM"], ["PCUM", "P5D"]],
    "P4 委托结构": [["NIMB", "SIZE"], ["NIMB", "SIZE", "DSIZE"], ["NIMB", "SIZE", "NIMB20"], ["NIMB", "SIZE", "DSIZE", "NIMB20"]],
    "P5 放量反转": [["HV"], ["HVJ"]],
    "P6 特质放量反转": [["IDHVJ"]],
}
for dn, vs in DIRS.items():
    for v in vs:
        x = np.mean([F[k] for k in v], axis=0)
        r = {wn: G.ev(x, wn) for wn in W}
        print("%-14s %-26s " % (dn, "+".join(v)) + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
