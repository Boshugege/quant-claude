# -*- coding: utf-8 -*-
"""可解释信号库第二批（10-20，"先扩知识面再优化"）：文献里机制清楚、现有分量没有覆盖的方向。
日内不变（只用前几个交易日）：
  K_MAX20    近 20 日最大日收益（Bali–Cakici–Whitelaw 2011：彩票偏好，散户追高 → 随后偏弱）
  K_IVOL20   近 20 日特质波动（日收益减等权市场收益的标准差；Ang et al. 2006 特质波动率之谜）
  K_ILLIQ20  Amihud 非流动性：近 20 日 |日收益| ÷ 成交额 的均值，取对数（Amihud 2002：流动性溢价）
  K_OVN20    近 20 日隔夜收益之和（Lou–Polk–Skouras 2019：隔夜与日内是两类投资者的拉锯）
  K_INTRA20  近 20 日日内收益（开盘到收盘）之和（同上，日内一侧）
  K_REV20    近 20 日累计收益（Jegadeesh 1990 月度反转）
  K_PVC20    近 20 日 日收益 与 log 成交额 的相关（Llorente et al. 2002：放量上涨是知情交易还是流动性冲击）
  K_SKEW20   近 20 日日收益偏度（Boyer–Mitton–Vorkink 2010：偏度偏好）
日内会变（当日到该截面为止）：
  I_GAP      今日开盘跳空：集合竞价价 ÷ 昨收 − 1（跳空回补）
  I_INTRA    今日开盘到该截面收盘的收益（Gao et al. 2018 日内动量）
保存 vc_tmp/h1_K_*.npy、h1_I_*.npy（原始值）。"""
import numpy as np
import pandas as pd
import root_eval as R
import h1_season as H1

t = pd.read_parquet(H1.CACHE); t["day"] = pd.to_datetime(t["day"])
W = H1.wide(t); del t
c15 = W["close"]["1500"]
auc = W["close"][H1.AUCT].where(W["close"][H1.AUCT] > 0, W["open"]["1000"])   # 集合竞价价；缺失时用首段开盘价
amt = W["amount"][H1.SEGS].sum(axis=1)
g = lambda s: s.groupby(level=0, sort=False)
clean = lambda s: s.replace([np.inf, -np.inf], np.nan)
pc = g(c15).shift(1)
ret = clean(c15 / pc - 1.0)
ovn = clean(auc / pc - 1.0)
intra = clean(c15 / auc - 1.0)
mkt = ret.groupby(level=1).transform("mean")
def roll(s, n, mp, fn):
    r = getattr(g(s).rolling(n, min_periods=mp), fn)()
    r.index = r.index.droplevel(0)
    return g(r.reindex(s.index)).shift(1)                                     # 严格只用前几日
K = {}
K["K_MAX20"] = roll(ret, 20, 10, "max")
K["K_IVOL20"] = roll(ret - mkt, 20, 10, "std")
K["K_ILLIQ20"] = np.log(roll(clean(ret.abs() / amt.where(amt > 0)), 20, 10, "mean") * 1e8 + 1e-6)
K["K_OVN20"] = roll(ovn, 20, 10, "sum")
K["K_INTRA20"] = roll(intra, 20, 10, "sum")
K["K_REV20"] = roll(np.log1p(ret), 20, 10, "sum")
la = np.log(amt.where(amt > 0))
mx, my = roll(ret, 20, 10, "mean"), roll(la, 20, 10, "mean")
cov = roll(ret * la, 20, 10, "mean") - mx * my
K["K_PVC20"] = cov / np.sqrt((roll(ret * ret, 20, 10, "mean") - mx * mx).clip(lower=1e-12) * (roll(la * la, 20, 10, "mean") - my * my).clip(lower=1e-12))
K["K_SKEW20"] = roll(ret, 20, 10, "skew")
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
def save(nm, fr):
    v = H1.align(c, fr); v[z930] = 0.0
    np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0))
    print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), flush=True)
for nm, ser in K.items():
    save(nm, pd.DataFrame({s0: clean(ser).to_numpy() for s0 in H1.CS}, index=ser.index))
save("I_GAP", pd.DataFrame({s0: ovn.to_numpy() for s0 in H1.CS}, index=ovn.index))
save("I_INTRA", pd.DataFrame({s0: clean(W["close"][s0] / auc - 1.0).to_numpy() for s0 in H1.CS}, index=auc.index))
