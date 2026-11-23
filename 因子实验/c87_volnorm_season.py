# -*- coding: utf-8 -*-
"""新信号源 N2 + N4（10-20 晚，用已算好的特征文件）：
N2 按波动率标准化的反转——同样 2% 的涨幅，在平时波动小的股票上是大冲击。σ = 该股过去约 20 日（140 个截面，错开 1 个）段收益的标准差：
  VN_SEG   −秩(D_R ÷ σ)          VN_IDIO  −秩((D_R − P_SEG) ÷ σ_idio)（σ_idio = 特质段收益的同样标准差）
N4 订单流的同时段季节性——同一股票同一时段过去 20 次的均值（错开 1 次）：
  SZ_OFI   特质主动买压 PD_MS_OFI     SZ_N23  深档笔数失衡 QL_NIMB23     SZ_SIZE  一档规模比 OQ_SIZE     SZ_VWAP  特质 VWAP 偏离 PD_AN_VWAP
单独五窗口五项；快信号另看平滑 α=0.5／0.6。"""
import numpy as np
import pandas as pd
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, MET, OUT
c = G.c; z = G.z930; did = c.date_id
W = ("2023", "2024", "2021", "2022", "2020")
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x):
    r = np.nan_to_num(O.rc_of(np.where(np.isfinite(x) & (x != 0), x, np.nan), did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
def show(tag, x):
    x = np.where(z, 0.0, x)
    r = {wn: G.ev(x, wn) for wn in W}
    print("%-22s " % tag + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
slot = c.slot_of_date[did]
order = np.lexsort((pd.to_datetime(c.key_date).values.astype(np.int64), c.inst))         # 按股票、时间排序
def roll_std(x, n=140, mp=40):
    s = pd.Series(np.where(z, np.nan, x)[order])
    r = s.groupby(c.inst[order]).transform(lambda q: q.rolling(n, min_periods=mp).std().shift(1)).to_numpy()
    out = np.empty_like(r); out[order] = r
    return out
def slot_mean(x, n=20, mp=8):
    o2 = np.lexsort((pd.to_datetime(c.key_date).values.astype(np.int64), slot, c.inst))
    s = pd.Series(np.where(z, np.nan, x)[o2])
    key = pd.MultiIndex.from_arrays([c.inst[o2], slot[o2]])
    r = s.groupby(key).transform(lambda q: q.rolling(n, min_periods=mp).mean().shift(1)).to_numpy()
    out = np.empty_like(r); out[o2] = r
    return out
DR, PS = raw("D_R"), raw("P_SEG")
DR = np.where(DR != 0, DR, np.nan); ID = np.where(PS != 0, DR - PS, np.nan)
sg, si = roll_std(DR), roll_std(ID)
cand = {"VN_SEG": -rk(DR / sg), "VN_IDIO": -rk(ID / si), "对照 特质段反转": -rk(ID)}
for nm, x in cand.items():
    show(nm, x)
    for a in (0.5, 0.6):
        show("%s α=%.1f" % (nm, a), H.ewma_col(x, c.inst, a))
for nm, src, sgn in (("SZ_OFI", "PD_MS_OFI", -1), ("SZ_N23", "QL_NIMB23", 1), ("SZ_SIZE", "OQ_SIZE", -1), ("SZ_VWAP", "PD_AN_VWAP", -1)):
    x = raw(src); x = np.where(x != 0, x, np.nan)
    sm = slot_mean(x)
    np.save(OUT / ("h1_%s.npy" % nm), np.nan_to_num(np.where(np.isfinite(sm) & (sm == 0), 1e-9, sm), nan=0.0))
    show(nm, sgn * rk(sm))
