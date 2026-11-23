# -*- coding: utf-8 -*-
"""纯方向深挖（10-20）：主题"流动性冲击回吐"内部组合——价格被非信息性的交易推开后被流动性提供者拉回。
快信号（截面秩取平均后按股票跨截面指数平滑 α；c70 显示平滑对快信号五项同时变好）：
  IDSEG  特质段反转  −秩(D_R − P_SEG)          IDHVJ 特质放量／跳跃反转  −秩(D_HV + D_JUMP − P_SEG)
  TICK   主动买压回吐 −秩(MS_OFI)               COFI  盘口订单流回吐  −秩(CF_OFI3)
慢信号（不平滑，直接加）：PCUM 特质日内反转 +秩(P_CUM)；OQ 委托结构（笔数失衡 + 规模比，α=0.3 平滑）；FLOW20 20 日订单流常态 +秩(CF_OFI3_20)
与参照池锚点（单机制因子，公榜 0.79～0.80）：2024 约 IC 0.068／IC_IR 1.17／多空 37／压力 1.10／换手 0.58。"""
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
PS = raw("P_SEG"); mP = mD & (PS != 0)
FAST = {"IDSEG": -rk(raw("D_R") - PS, mP), "IDHVJ": -rk(raw("D_HV") + raw("D_JUMP") - PS, mP),
        "TICK": -rk(raw("MS_OFI")), "COFI": -rk(raw("CF_OFI3"))}
PCUM = rk(raw("P_CUM"))
OQ = H.ewma_col(np.mean([rk(raw("OQ_NIMB")), -rk(raw("OQ_SIZE"))], axis=0), c.inst, 0.3)
FLOW20 = rk(raw("CF_OFI3_20"))
def show(tag, x):
    x = np.where(z, 0.0, x)
    r = {wn: G.ev(x, wn) for wn in W}
    print("%-44s " % tag + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
SETS = [["IDSEG"], ["IDSEG", "IDHVJ"], ["IDSEG", "TICK"], ["IDSEG", "COFI"], ["IDSEG", "IDHVJ", "TICK"], ["IDSEG", "IDHVJ", "TICK", "COFI"]]
for a in (0.5, 0.6, 0.7):
    for st in SETS:
        f = H.ewma_col(np.mean([FAST[k] for k in st], axis=0), c.inst, a)
        show("α=%.1f %s" % (a, "+".join(st)), f)
        for wp in (0.25, 0.5):
            show("α=%.1f %s + %.2f·PCUM" % (a, "+".join(st), wp), f + wp * PCUM)
        show("α=%.1f %s + 0.25·PCUM + 0.25·OQ" % (a, "+".join(st)), f + 0.25 * PCUM + 0.25 * OQ)
        show("α=%.1f %s + 0.25·PCUM + 0.25·OQ + 0.25·FLOW20" % (a, "+".join(st)), f + 0.25 * PCUM + 0.25 * OQ + 0.25 * FLOW20)
