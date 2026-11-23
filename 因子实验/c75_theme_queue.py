# -*- coding: utf-8 -*-
"""纯方向深挖（10-20）：主题"委托结构／排队意愿"内部组合——很多小单排在买方（尤其后排档位）= 真实需求，少数大单挂着 = 压盘。
快信号（截面秩取平均后按股票跨截面指数平滑 α）：N23 第 2、3 档笔数失衡（+）、NA 1～3 档笔数失衡（+）、N1 一档笔数失衡（+）、
  S1 一档单笔规模比（−）、SD 1～3 档单笔规模比（−）、SL 段内最后 10 分钟规模比（−）
慢信号（不平滑）：NA20 1～3 档笔数失衡 20 日常态（+）、F20 1～3 档盘口订单流 20 日常态（+）
与参照池锚点（单机制因子，公榜 0.79～0.80）：2024 约 IC 0.068／IC_IR 1.17／多空 37／压力 1.10／换手 0.58。"""
import numpy as np
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, MET, OUT
c = G.c; z = G.z930; did = c.date_id
W = ("2023", "2024", "2021", "2022", "2020")
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x):
    r = np.nan_to_num(O.rc_of(np.where(x != 0, x, np.nan), did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
FAST = {"N23": rk(raw("QL_NIMB23")), "NA": rk(raw("QL_NIMBA")), "N1": rk(raw("OQ_NIMB")),
        "S1": -rk(raw("OQ_SIZE")), "SD": -rk(raw("OQ_DSIZE")), "SL": -rk(raw("QL_SLATE"))}
NA20, F20 = rk(raw("QL_NIMBA_20")), rk(raw("CF_OFI3_20"))
def show(tag, x):
    x = np.where(z, 0.0, x)
    r = {wn: G.ev(x, wn) for wn in W}
    print("%-40s " % tag + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
SETS = [["N23"], ["NA"], ["N23", "S1"], ["NA", "S1"], ["N23", "S1", "SL"], ["NA", "S1", "SD"], ["N23", "N1", "S1", "SL"]]
for a in (0.0, 0.3, 0.5, 0.7):
    for st in SETS:
        f = np.mean([FAST[k] for k in st], axis=0)
        f = H.ewma_col(f, c.inst, a) if a > 0 else f
        show("α=%.1f %s" % (a, "+".join(st)), f)
        if len(st) > 1:
            show("α=%.1f %s + 0.25·NA20" % (a, "+".join(st)), f + 0.25 * NA20)
            show("α=%.1f %s + 0.25·NA20 + 0.25·F20" % (a, "+".join(st)), f + 0.25 * NA20 + 0.25 * F20)
