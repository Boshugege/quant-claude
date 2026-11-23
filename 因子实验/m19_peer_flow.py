# -*- coding: utf-8 -*-
"""纯方向深挖 J7（10-20）：同伴的订单流与委托结构（领先—滞后在订单流层面的版本；以前只在收益层面测过，m11）。
同伴 = 每天用严格过去 60 个交易日的日收益相关，取最相关的 20 只（同 m11），权重 = 相关系数（只在同一截面有值的同伴上归一）。
对每个信号 X ∈ {QL_NIMB23 深档笔数失衡, OQ_SIZE 一档单笔规模比, MS_OFI 主动买压, CF_OFI3 盘口订单流}：
  PF_X   同伴在同一截面的加权均值
  PD_X   本股 − 同伴（相对同伴的偏离）
保存 vc_tmp/h1_PF_*.npy、h1_PD_*.npy（原始值）。"""
import numpy as np
import pandas as pd
import root_eval as R
import h1_season as H1

LB, K, MINN = 60, 20, 40
import os
SIGS = os.environ.get("SIGS", "QL_NIMB23,OQ_SIZE,MS_OFI,CF_OFI3").split(",")              # 可用环境变量换信号（10-20 起也算 AN_VWAP 等）
t = pd.read_parquet(H1.CACHE); t["day"] = pd.to_datetime(t["day"])
W = H1.wide(t); del t
c15 = W["close"]["1500"]
ret = (c15 / c15.groupby(level=0, sort=False).shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
RD = ret.unstack(0)                                                            # 日 × 股票
days, ids = RD.index, RD.columns.to_numpy(np.int64)
A = RD.to_numpy(float)
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
X = {s: np.load(H1.OUT / ("h1_%s.npy" % s)).astype(float) for s in SIGS}
for s in SIGS:
    X[s][X[s] == 0] = np.nan
rday = pd.to_datetime(c.day_of_date[c.date_id]).values
rcol = pd.Index(ids).get_indexer(c.inst.astype(np.int64))
dpos = pd.Index(days).get_indexer(rday)
order = np.argsort(c.date_id, kind="stable")
sec = c.date_id[order]
bounds = np.flatnonzero(np.r_[True, sec[1:] != sec[:-1], True])
out = {("PF_" + s): np.full(c.n_rows, np.nan) for s in SIGS}
cache_day, nb, wv, live = -1, None, None, None
for bi in range(len(bounds) - 1):
    rows = order[bounds[bi]:bounds[bi + 1]]
    if z930[rows[0]]:
        continue
    i = dpos[rows[0]]
    if i < LB:
        continue
    if i != cache_day:                                                         # 当日同伴（只用前 LB 日）
        hist = A[i - LB:i]
        live = np.flatnonzero((np.isfinite(hist).sum(0) >= MINN) & np.isfinite(A[i - 1]))
        if len(live) < 3 * K:
            cache_day, nb = i, None
            continue
        h = hist[:, live]
        Z = np.nan_to_num((h - np.nanmean(h, 0)) / np.where(np.nanstd(h, 0) > 0, np.nanstd(h, 0), np.nan))
        cor = Z.T @ Z / LB
        np.fill_diagonal(cor, -np.inf)
        nb = np.argpartition(-cor, K, axis=1)[:, :K]
        wv = np.take_along_axis(cor, nb, 1).clip(min=0.0)
        cache_day = i
    if nb is None:
        continue
    pos = pd.Index(live).get_indexer(rcol[rows])                               # 截面行 → 当日同伴矩阵的位置
    okr = pos >= 0
    for s in SIGS:
        v = np.full(len(live), np.nan)
        v[pos[okr]] = X[s][rows[okr]]
        vn = v[nb]
        w = np.where(np.isfinite(vn), wv, 0.0)
        pf = np.nansum(w * np.nan_to_num(vn), 1) / np.where(w.sum(1) > 0, w.sum(1), np.nan)
        out["PF_" + s][rows[okr]] = pf[pos[okr]]
    if bi % 2000 == 0:
        print("截面", bi, len(bounds) - 1, flush=True)
for s in SIGS:
    pf = out["PF_" + s]
    for nm, v in (("PF_" + s, pf), ("PD_" + s, X[s] - pf)):
        v = np.where(np.isfinite(v) & (v == 0), 1e-9, v); v[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), flush=True)
