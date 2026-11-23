# -*- coding: utf-8 -*-
"""可解释信号库第四批（10-20）：同伴领先—滞后（Hou 2007；Cohen–Frazzini 2008；Ali–Hirshleifer 2020）。
同伴 = 每天用严格过去 60 个交易日的日收益相关，取最相关的 20 只（只用行情数据，不用行业表）；权重 = 相关系数。
  P_SEG    同伴在当前半小时段的加权收益（同伴刚涨 → 本股随后跟涨）
  P_LAG    P_SEG − 本股当前段收益（相对同伴落后多少，含本股自身的反转成分）
  P_CUM    同伴今日开盘至截面的收益 − 本股同期收益
  P_DAY    同伴昨日收益 − 本股昨日收益（日频，日内不变）
保存 vc_tmp/h1_P_*.npy（原始值）。"""
import numpy as np
import pandas as pd
import root_eval as R
import h1_season as H1

LB, K, MINN = 60, 20, 40
t = pd.read_parquet(H1.CACHE); t["day"] = pd.to_datetime(t["day"])
W = H1.wide(t); del t
C = W["close"]
c15 = C["1500"]
auc = C[H1.AUCT].where(C[H1.AUCT] > 0, W["open"]["1000"])
g = lambda s: s.groupby(level=0, sort=False)
clean = lambda s: s.replace([np.inf, -np.inf], np.nan)
ret = clean(c15 / g(c15).shift(1) - 1.0)
prev = {"1000": None, "1030": "1000", "1100": "1030", "1130": "1100", "1330": "1130", "1400": "1330", "1430": "1400"}
seg = pd.DataFrame({s0: clean(C[s0] / (C[p] if p else auc) - 1.0) for s0, p in prev.items()})
cum = pd.DataFrame({s0: clean(C[s0] / auc - 1.0) for s0 in H1.CS})
RD = ret.unstack(0)                                                            # 日 × 股票
days = RD.index
SEG = {s0: seg[s0].unstack(0).reindex(index=days, columns=RD.columns) for s0 in H1.CS}
CUM = {s0: cum[s0].unstack(0).reindex(index=days, columns=RD.columns) for s0 in H1.CS}
out = {nm: np.full((len(days), RD.shape[1], len(H1.CS)), np.nan) for nm in ("P_SEG", "P_LAG", "P_CUM")}
pday = np.full((len(days), RD.shape[1]), np.nan)
A = RD.to_numpy(float)
for i in range(LB, len(days)):
    hist = A[i - LB:i]
    live = np.flatnonzero((np.isfinite(hist).sum(0) >= MINN) & np.isfinite(A[i - 1]) & np.isfinite(SEG["1000"].iloc[i].to_numpy(float)))
    if len(live) < 3 * K:
        continue
    h = hist[:, live]
    mu = np.nanmean(h, 0); sd = np.nanstd(h, 0)
    Z = np.nan_to_num((h - mu) / np.where(sd > 0, sd, np.nan))
    cor = Z.T @ Z / LB
    np.fill_diagonal(cor, -np.inf)
    nb = np.argpartition(-cor, K, axis=1)[:, :K]                               # 每只股票最相关的 K 只
    wv = np.take_along_axis(cor, nb, 1).clip(min=0.0)
    wv = wv / np.where(wv.sum(1, keepdims=True) > 0, wv.sum(1, keepdims=True), np.nan)
    yd = A[i - 1, live]
    pday[i, live] = np.nansum(wv * yd[nb], 1) - yd
    for j, s0 in enumerate(H1.CS):
        x = SEG[s0].iloc[i].to_numpy(float)[live]
        y = CUM[s0].iloc[i].to_numpy(float)[live]
        ps = np.nansum(wv * np.nan_to_num(x)[nb], 1)
        out["P_SEG"][i, live, j] = ps
        out["P_LAG"][i, live, j] = ps - x
        out["P_CUM"][i, live, j] = np.nansum(wv * np.nan_to_num(y)[nb], 1) - y
    if i % 200 == 0:
        print("日", i, len(days), "股票", len(live), flush=True)
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
idx = pd.MultiIndex.from_product([RD.columns, days], names=["iid", "day"])
def save(nm, arr3):                                                            # arr3：日 × 股票 × 截面
    fr = pd.DataFrame({s0: arr3[:, :, j].T.reshape(-1) for j, s0 in enumerate(H1.CS)}, index=idx)
    v = H1.align(c, fr); v[z930] = 0.0
    np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0))
    print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), flush=True)
for nm, arr in out.items():
    save(nm, arr)
save("P_DAY", np.repeat(pday[:, :, None], len(H1.CS), 2))
