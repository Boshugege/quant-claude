# -*- coding: utf-8 -*-
"""R1 探索候选：日线尺度反转、集合竞价、日内节奏、波动结构等新方向（30 分钟面板可实现的部分）。

每个函数接收已 add_components 的面板 p，返回与 p 等长的因子值（只用过去数据）。
"""
import numpy as np
import pandas as pd

from factor_lib import grolling, _log_ratio, rank_blend, ladder


# ---------------------------------------------------------------- 工具
def _daily(p):
    d = (p.groupby(["instrument_id", "day"], sort=False)
           .agg(adj_last=("adj_px", "last"), pc=("pre_close", "first"),
                amt=("amount_c", "sum"), vol=("volume_c", "sum"),
                px_open=("open", "first"))
           .reset_index().sort_values(["instrument_id", "day"]))
    return d


def _merge_daily(p, d, col):
    m = p[["instrument_id", "day"]].merge(d[["instrument_id", "day", col]], on=["instrument_id", "day"], how="left")[col]
    m.index = p.index
    return m


def _win_ret(p):
    return _log_ratio(p["px"], p.groupby("instrument_id", sort=False)["px"].shift(1))


# ---------------------------------------------------------------- 1. 日线尺度反转谱
def _day_lag_close(p, k):
    d = _daily(p)
    d["base"] = d.groupby("instrument_id", sort=False)["adj_last"].shift(k)
    return _merge_daily(p, d, "base")


def d5_rev(p):
    """5 个交易日的日线尺度反转：ln(5 日前复权收盘 / 当前价)。"""
    return _log_ratio(_day_lag_close(p, 5), p["adj_px"])


def d10_rev(p):
    return _log_ratio(_day_lag_close(p, 10), p["adj_px"])


def d20_rev(p):
    """约 20 个交易日（一个月）的日线尺度反转。端到端多尺度专家加了这一尺度后 IC_IR、压力双升。"""
    return _log_ratio(_day_lag_close(p, 20), p["adj_px"])


# ---------------------------------------------------------------- 2. 集合竞价
def auc_share(p):
    """集合竞价成交额 / 昨日全天成交额（情绪强度、关注度）。"""
    d = _daily(p)
    d["prev_amt"] = d.groupby("instrument_id", sort=False)["amt"].shift(1)
    a = (p["amount_c"].where(p["slot"] == 930, 0.0)
         .groupby([p["instrument_id"], p["day"]]).sum().rename("auc_amt").reset_index())
    d = d.merge(a, on=["instrument_id", "day"], how="left")
    v = d["auc_amt"] / d["prev_amt"].where(d["prev_amt"] > 0)
    d["v"] = np.log(v.where(v > 0))
    return _merge_daily(p, d, "v")


def auc_share_hist(p):
    """集合竞价成交额占昨日成交额的比例，相对过去 10 日均值的偏离。"""
    d = _daily(p)
    d["prev_amt"] = d.groupby("instrument_id", sort=False)["amt"].shift(1)
    a = (p["amount_c"].where(p["slot"] == 930, 0.0)
         .groupby([p["instrument_id"], p["day"]]).sum().rename("auc_amt").reset_index())
    d = d.merge(a, on=["instrument_id", "day"], how="left")
    r = d["auc_amt"] / d["prev_amt"].where(d["prev_amt"] > 0)
    d["r"] = np.log(r.where(r > 0))
    d = d.sort_values(["instrument_id", "day"])
    d["v"] = d["r"] - d.groupby("instrument_id", sort=False)["r"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    return _merge_daily(p, d, "v")


# ---------------------------------------------------------------- 3. 日内节奏
def vol_centroid(p):
    """当日成交金额的时间重心（0=开盘，1=收盘）相对过去 10 日同一比值的偏离。"""
    seq = p.groupby(["instrument_id", "day"], sort=False).cumcount().astype(float)
    nmax = p.groupby(["instrument_id", "day"], sort=False)["day"].transform("size").astype(float).sub(1.0)
    pos = seq / nmax.where(nmax > 0)
    w = p["amount_c"].fillna(0.0)
    num = (w * pos).groupby([p["instrument_id"], p["day"]]).sum().rename("num").reset_index()
    den = w.groupby([p["instrument_id"], p["day"]]).sum().rename("den").reset_index()
    d = num.merge(den, on=["instrument_id", "day"])
    d["c"] = d["num"] / d["den"].where(d["den"] > 0)
    d = d.sort_values(["instrument_id", "day"])
    d["hist"] = d.groupby("instrument_id", sort=False)["c"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    d["v"] = d["c"] - d["hist"]
    return _merge_daily(p, d, "v")


def tail_share_dev(p):
    """当日累计成交额中「最近两个窗口」的占比（尾盘集中度）相对历史同时点的偏离。"""
    seq = p.groupby(["instrument_id", "day"], sort=False).cumcount()
    last2 = p["amount_c"].where(seq >= 2, 0.0)
    num = last2.groupby([p["instrument_id"], p["day"]]).cumsum()
    den = p["amount_c"].groupby([p["instrument_id"], p["day"]]).cumsum()
    r = (num / den.where(den > 0))
    hist = r.groupby([p["instrument_id"], p["slot"]], sort=False).transform(
        lambda x: x.shift(1).rolling(10, min_periods=3).mean())
    return r - hist


# ---------------------------------------------------------------- 4. 波动 / 流动性结构
def downside_share(p):
    """下行波动占比：最近 40 窗口负收益平方和 / 全部收益平方和（高占比=下行风险大）。"""
    r = _win_ret(p)
    neg2 = r.where(r < 0, 0.0) ** 2
    sn = grolling(neg2, p["instrument_id"], 40, "sum", 8)
    st = grolling(r ** 2, p["instrument_id"], 40, "sum", 8)
    return sn / st.where(st > 0)


def vol_chg(p):
    """波动率变化：最近 10 窗口已实现波动 / 最近 40 窗口（>0 表示波动放大）。"""
    r = _win_ret(p)
    s10 = grolling(r, p["instrument_id"], 10, "std", 5)
    s40 = grolling(r, p["instrument_id"], 40, "std", 15)
    v = s10 / s40.where(s40 > 0)
    return np.log(v.where(v > 0))


def illiq_chg(p):
    """非流动性的近期相对水平：最近 8 窗口 Amihud / 最近 40 窗口 Amihud。"""
    il = _win_ret(p).abs() / (p["amount_c"] / 1e8).where(p["amount_c"] > 0)
    s8 = grolling(il, p["instrument_id"], 8, "mean", 4)
    s40 = grolling(il, p["instrument_id"], 40, "mean", 10)
    v = s8 / s40.where(s40 > 0)
    return np.log(v.where(v > 0))


def avg_trade_lvl(p):
    """当日至今平均单笔成交额（元）的对数：区分大单主导与散户小单主导。"""
    v = p["avg_trade"].where(p["avg_trade"] > 0)
    return np.log(v)


# ---------------------------------------------------------------- 5. 日内动量
def first30_mom(p):
    """首半小时动量：10:00 价格相对昨收的收益（Gao 等 2018 的日内动量所在时点）。"""
    d = _daily(p)
    f = p.loc[p["slot"] == 1000, ["instrument_id", "day", "px"]].rename(columns={"px": "px1000"})
    d = d.merge(f, on=["instrument_id", "day"], how="left")
    d["v"] = _log_ratio(d["px1000"], d["pc"])
    return _merge_daily(p, d, "v")


# ---------------------------------------------------------------- 6. 与主阶梯的融合
def d20_into_ladder(p):
    """20 日反转并入四尺度阶梯（等权）。"""
    cols = {"a": ladder(p, ["30", "rv4", "rv8", "day"], False), "b": d20_rev(p)}
    return rank_blend(p.assign(**cols), ["a", "b"])


def d20_into_ladder_half(p):
    """20 日反转以半权重并入四尺度阶梯。"""
    cols = {"a": ladder(p, ["30", "rv4", "rv8", "day"], False), "b": d20_rev(p)}
    return rank_blend(p.assign(**cols), ["a", "b"], [1.0, 0.5])


CANDS = {
    "d5_rev": d5_rev,
    "d10_rev": d10_rev,
    "d20_rev": d20_rev,
    "auc_share": auc_share,
    "auc_share_hist": auc_share_hist,
    "vol_centroid": vol_centroid,
    "tail_share_dev": tail_share_dev,
    "downside_share": downside_share,
    "vol_chg": vol_chg,
    "illiq_chg": illiq_chg,
    "avg_trade_lvl": avg_trade_lvl,
    "first30_mom": first30_mom,
    "d20_into_ladder": d20_into_ladder,
    "d20_into_ladder_half": d20_into_ladder_half,
}
