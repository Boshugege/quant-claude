# -*- coding: utf-8 -*-
"""方向 C（10-20）：特质反转阶梯的原料。同伴定义同 m11（严格过去 60 日日收益相关，取最相关的 20 只，权重 = 相关系数）。
  P_GAP   同伴今日隔夜跳空 − 本股（集合竞价价 ÷ 昨收 − 1）
  P_5D    同伴近 5 日累计收益 − 本股（第 d−5..d−1 日，日内不变）
  P_20D   同伴近 20 日累计收益 − 本股
  O_5D    本股近 5 日累计收益（对照：不扣同伴）
保存 vc_tmp/h1_P_*.npy、h1_O_5D.npy（原始值）。"""
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
lr = np.log1p(clean(c15 / g(c15).shift(1) - 1.0))
gap = clean(auc / g(c15).shift(1) - 1.0)
RD = lr.unstack(0)
days = RD.index
A = RD.to_numpy(float)
GP = gap.unstack(0).reindex(index=days, columns=RD.columns).to_numpy(float)
cs = np.vstack([np.zeros((1, A.shape[1])), np.nancumsum(np.nan_to_num(A), 0)])   # cs[i] = 前 i 日之和
cnt = np.vstack([np.zeros((1, A.shape[1])), np.cumsum(np.isfinite(A), 0)])
def past(i, h):                                                               # 第 i−h..i−1 日累计，有效天数不足一半记缺失
    s, n = cs[i] - cs[i - h], cnt[i] - cnt[i - h]
    return np.where(n >= h / 2, s, np.nan)
out = {nm: np.full(A.shape, np.nan) for nm in ("P_GAP", "P_5D", "P_20D", "O_5D")}
for i in range(LB, len(days)):
    hist = A[i - LB:i]
    live = np.flatnonzero((np.isfinite(hist).sum(0) >= MINN) & np.isfinite(A[i - 1]) & np.isfinite(GP[i]))
    if len(live) < 3 * K:
        continue
    h = hist[:, live]
    mu = np.nanmean(h, 0); sd = np.nanstd(h, 0)
    Z = np.nan_to_num((h - mu) / np.where(sd > 0, sd, np.nan))
    cor = Z.T @ Z / LB
    np.fill_diagonal(cor, -np.inf)
    nb = np.argpartition(-cor, K, axis=1)[:, :K]
    wv = np.take_along_axis(cor, nb, 1).clip(min=0.0)
    wv = wv / np.where(wv.sum(1, keepdims=True) > 0, wv.sum(1, keepdims=True), np.nan)
    pe = lambda x: np.nansum(wv * np.nan_to_num(x)[nb], 1) - x               # noqa: E731
    out["P_GAP"][i, live] = pe(GP[i, live])
    x5 = past(i, 5)[live]
    out["P_5D"][i, live] = pe(x5)
    out["P_20D"][i, live] = pe(past(i, 20)[live])
    out["O_5D"][i, live] = x5
    if i % 300 == 0:
        print("日", i, len(days), flush=True)
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
idx = pd.MultiIndex.from_product([RD.columns, days], names=["iid", "day"])
for nm, arr in out.items():
    v1 = arr.T.reshape(-1)
    fr = pd.DataFrame({s0: v1 for s0 in H1.CS}, index=idx)
    v = H1.align(c, fr); v[z930] = 0.0
    np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0))
    print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), flush=True)
