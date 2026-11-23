# -*- coding: utf-8 -*-
"""可解释信号库第一批（10-19 夜，"先扩知识面再优化"）：都是日内不变的个股特征，只用前几个交易日的数据（当日各截面相同）。
  K_PH250  52 周高点锚定：昨收 ÷ 过去 250 日最高收盘（George–Hwang 2004：接近高点时对利好反应不足 → 延续）
  K_VTS    波动率期限结构：近 5 日日收益标准差 ÷ 近 60 日（短期波动突增 → 过度反应）
  K_VOLTS  成交量期限结构：近 5 日日均成交额 ÷ 近 60 日（放量吸引注意力 → 随后回吐，Barber–Odean 2008）
  K_MDD20  近 20 日最大回撤（恐慌抛售后的修复）
  K_CGO    资本利得悬置：昨收 ÷ 近 60 日成交额加权均价 − 1（处置效应压制浮盈股，Grinblatt–Han 2005）
保存 vc_tmp/h1_K_*.npy（原始值）。"""
import numpy as np
import pandas as pd
import root_eval as R
import h1_season as H1

t = pd.read_parquet(H1.CACHE); t["day"] = pd.to_datetime(t["day"])
W = H1.wide(t); del t
c15 = W["close"]["1500"]
amt = W["amount"][H1.SEGS].sum(axis=1)
vol = W["volume"][H1.SEGS].sum(axis=1)
vwap = (amt / vol.where(vol > 0))
g = lambda s: s.groupby(level=0, sort=False)
ret = (c15 / g(c15).shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
def roll(s, n, mp, fn):
    r = getattr(g(s).rolling(n, min_periods=mp), fn)()
    r.index = r.index.droplevel(0)
    return g(r.reindex(s.index)).shift(1)                                     # 严格只用前几日
K = {}
K["K_PH250"] = g(c15).shift(1) / roll(c15, 250, 120, "max")
K["K_VTS"] = roll(ret, 5, 4, "std") / roll(ret, 60, 30, "std")
K["K_VOLTS"] = roll(amt, 5, 4, "mean") / roll(amt, 60, 30, "mean")
dd = c15 / g(c15).cummax() - 1.0                                           # 相对历史最高（逐股）——近 20 日最大回撤用滚动窗内的相对最高
runmax20 = g(c15).rolling(20, min_periods=10).max(); runmax20.index = runmax20.index.droplevel(0); runmax20 = runmax20.reindex(c15.index)
K["K_MDD20"] = g(c15 / runmax20 - 1.0).rolling(20, min_periods=10).min().droplevel(0).reindex(c15.index).groupby(level=0, sort=False).shift(1)
pv = roll(vwap * amt, 60, 30, "sum") / roll(amt, 60, 30, "sum")
K["K_CGO"] = g(c15).shift(1) / pv - 1.0
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
for nm, ser in K.items():
    fr = pd.DataFrame({s0: ser.replace([np.inf, -np.inf], np.nan).to_numpy() for s0 in H1.CS}, index=ser.index)
    v = H1.align(c, fr); v[z930] = 0.0
    np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0))
    print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), flush=True)
