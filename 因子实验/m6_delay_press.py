# -*- coding: utf-8 -*-
"""有经济含义的新根（10-19 傍晚）：
A 价格延迟（Hou & Moskowitz 2005）：个股 30 分钟段收益对【同段】与【上一段】市场收益回归（过去 20 日、每日 8 段，只用前几日），
  滞后系数 b1 大 = 对市场信息反应慢。PD = b1 × 刚过去这一段的市场收益（预期个股下一段跟上市场）。另存 b1 本身（LAGB20）。
C 流动性冲击回归：PRESS = 最近一段个股收益 × 该段深度消耗率（BK_CONS 截面秩 + 0.5），大 = 吃盘口推出来的价格变动（预期回吐）。
保存 vc_tmp/h1_PD.npy、h1_LAGB20.npy、h1_PRESS.npy（原始值）。"""
import numpy as np
import pandas as pd
import root_eval as R
import d2_optimize as O
import h1_season as H1

t = pd.read_parquet(H1.CACHE); t["day"] = pd.to_datetime(t["day"])
W = H1.wide(t); del t
close, open_ = W["close"], W["open"]
S = H1.SEGS
prev = close[S].shift(1, axis=1); prev["1000"] = open_["1000"]
y = (close[S] / prev - 1.0).replace([np.inf, -np.inf], np.nan)
day = y.index.get_level_values(1)
mk = {}
for s0 in S:
    g = pd.DataFrame({"day": day, "v": y[s0].to_numpy(float)})
    q = g.groupby("day")["v"].quantile([0.01, 0.99]).unstack(); g = g.join(q, on="day")
    mk[s0] = g.assign(v=g["v"].clip(g[0.01], g[0.99])).groupby("day")["v"].mean()
M = pd.DataFrame(mk).sort_index()                                          # day × 段
Mlag = M.shift(1, axis=1); Mlag["1000"] = M["1500"].shift(1)                # 上一段（每日第一段取前一日最后一段）
x0 = pd.DataFrame(M.reindex(day).to_numpy(), index=y.index, columns=S)
x1 = pd.DataFrame(Mlag.reindex(day).to_numpy(), index=y.index, columns=S)
ok = y.notna() & x0.notna() & x1.notna()
yv, a, b = y.where(ok), x0.where(ok), x1.where(ok)
agg = pd.DataFrame({"n": ok.sum(1), "sa": a.sum(1), "sb": b.sum(1), "sy": yv.sum(1), "saa": (a * a).sum(1), "sbb": (b * b).sum(1),
                    "sab": (a * b).sum(1), "say": (a * yv).sum(1), "sby": (b * yv).sum(1)})
roll = agg.groupby(level=0, sort=False).rolling(20, min_periods=10).sum(); roll.index = roll.index.droplevel(0)
roll = roll.reindex(agg.index).groupby(level=0, sort=False).shift(1)       # 严格只用前几日
n = roll["n"].where(roll["n"] >= 80)
caa = roll["saa"] - roll["sa"] ** 2 / n; cbb = roll["sbb"] - roll["sb"] ** 2 / n; cab = roll["sab"] - roll["sa"] * roll["sb"] / n
cay = roll["say"] - roll["sa"] * roll["sy"] / n; cby = roll["sby"] - roll["sb"] * roll["sy"] / n
det = caa * cbb - cab * cab
b1 = ((caa * cby - cab * cay) / det).where(det > 0)                          # 二元回归的滞后系数
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
LB = pd.DataFrame({s0: b1.to_numpy() for s0 in H1.CS}, index=y.index)
PD = pd.DataFrame({s0: b1.to_numpy() * M[s0].reindex(day).to_numpy() for s0 in H1.CS}, index=y.index)   # 段 s0 = (t−30, t] 的市场收益
for nm, fr in (("LAGB20", LB), ("PD", PD)):
    v = H1.align(c, fr); v[z930] = 0.0
    np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0)); print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), flush=True)
ry = H1.align(c, y[H1.CS]); ry[z930] = np.nan                               # 最近一段个股收益
cons = np.load(H1.OUT / "h1_BK_CONS.npy").astype(float); cons[cons == 0] = np.nan
rc = O.rc_of(cons, c.date_id, c.n_dates) + 0.5
pr = ry * rc; pr[z930] = 0.0
np.save(H1.OUT / "h1_PRESS.npy", np.nan_to_num(pr, nan=0.0)); print("保存 PRESS 非零率 %.3f" % float((np.nan_to_num(pr) != 0).mean()), flush=True)
np.save(H1.OUT / "h1_RSEG.npy", np.nan_to_num(np.where(z930, 0.0, ry), nan=0.0)); print("保存 RSEG（对照：最近一段收益本身）", flush=True)
