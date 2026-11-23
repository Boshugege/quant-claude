# -*- coding: utf-8 -*-
"""R6：四个从未使用过的窗口内结构信号（价格路径不对称、30 分钟收益自相关、量价背离、VWAP 位置趋势）。"""
import numpy as np
import pandas as pd

import factor_lib as fl
from factor_lib import _log_ratio, grolling, neutral4, rank_blend
from cand_r3 import base_104, base_soft2, blend


def path_asym(p):
    """窗口收盘价相对本窗口成交 VWAP 的位置（买压/卖压不对称），最近 40 个窗口的均值。"""
    v = _log_ratio(p["px"], p["vwap_w"])
    return grolling(v, p["instrument_id"], 40, "mean", 10)


def _lag1_ret(p):
    r = _log_ratio(p["px"], p.groupby("instrument_id", sort=False)["px"].shift(1))
    return r, r.groupby(p["instrument_id"], sort=False).shift(1)


def ac30(p):
    """30 分钟窗口收益的一阶自相关（最近 40 个窗口）：负值大 = 窗口间彼此反转。"""
    r, r1 = _lag1_ret(p)
    i, x, y = p["instrument_id"], r1, r
    mx = grolling(x, i, 40, "mean", 10)
    my = grolling(y, i, 40, "mean", 10)
    mxy = grolling(x * y, i, 40, "mean", 10)
    vx = (grolling(x * x, i, 40, "mean", 10) - mx ** 2).clip(lower=0)
    vy = (grolling(y * y, i, 40, "mean", 10) - my ** 2).clip(lower=0)
    return (mxy - mx * my) / np.sqrt(vx * vy).replace(0.0, np.nan)


def vol_price_div(p):
    """窗口收益与成交额对数变化的截面相关性（最近 40 个窗口）：量价同向 = 追涨杀跌。"""
    r, _ = _lag1_ret(p)
    a = np.log(p["amount_c"].where(p["amount_c"] > 0))
    da = a - a.groupby(p["instrument_id"], sort=False).shift(1)
    i = p["instrument_id"]
    mr, md = grolling(r, i, 40, "mean", 10), grolling(da, i, 40, "mean", 10)
    mrd = grolling(r * da, i, 40, "mean", 10)
    vr = (grolling(r * r, i, 40, "mean", 10) - mr ** 2).clip(lower=0)
    vd = (grolling(da * da, i, 40, "mean", 10) - md ** 2).clip(lower=0)
    return (mrd - mr * md) / np.sqrt(vr * vd).replace(0.0, np.nan)


def vwap_pos_trend(p):
    """窗口 VWAP 相对当日开盘价的偏离与其 4 窗口变化（日内价格重心的移动）。"""
    pos = _log_ratio(p["vwap_w"], p["open"])
    chg = pos - pos.groupby(p["instrument_id"], sort=False).shift(4)
    return chg


def close_path_skew(p):
    """窗口内 (收盘 - VWAP) 的偏度（最近 20 个窗口），取负。"""
    v = _log_ratio(p["px"], p["vwap_w"])
    m = grolling(v, p["instrument_id"], 20, "mean", 8)
    s = grolling(v, p["instrument_id"], 20, "std", 8)
    m3 = grolling(v ** 3, p["instrument_id"], 20, "mean", 8)
    return -m3 / (s ** 3).replace(0.0, np.nan)


CANDS = {
    "path_asym": path_asym,
    "ac30": ac30,
    "vol_price_div": vol_price_div,
    "vwap_pos_trend": vwap_pos_trend,
    "close_path_skew": close_path_skew,
    "104_path_asym": blend(base_104, path_asym, 0.15),
    "104_ac30": blend(base_104, ac30, 0.15),
    "104_vpd": blend(base_104, vol_price_div, 0.15),
    "104_vpt": blend(base_104, vwap_pos_trend, 0.15),
    "soft2_path_asym": blend(base_soft2, path_asym, 0.15),
    "soft2_ac30": blend(base_soft2, ac30, 0.15),
    "soft2_vpd": blend(base_soft2, vol_price_div, 0.15),
}
