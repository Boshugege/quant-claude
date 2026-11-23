# -*- coding: utf-8 -*-
"""R3：R2 赢家的组合与权重扫描，以及新方向（成交集中度、特质偏度、跳空、振幅、尾部损失）。"""
import numpy as np
import pandas as pd

import factor_lib as fl
from factor_lib import (amihud, grolling, ivol, max_effect, neutral4, neutralize,
                        rank_blend, size_neg, _log_ratio)
from cand_r1 import d5_rev, downside_share, vol_centroid
from cand_r2 import peer_res_ladder_w


def base_104(p):
    return neutral4(p, fl.peer_res_ladder(p), 1.5)


def base_soft2(p):
    return neutral4(p, peer_res_ladder_w(p, mode="soft", power=2.0, k=100), 1.5)


def base_soft1(p):
    return neutral4(p, peer_res_ladder_w(p, mode="soft", power=1.0, k=100), 1.5)


def blend(base_fn, add_fn, w):
    def f(p):
        return rank_blend(p.assign(a=base_fn(p), b=add_fn(p)), ["a", "b"], [1.0, w])
    return f


# ---------------------------------------------------------------- 新方向
def n_herf(p):
    """当日成交金额在已发生窗口间的集中度（Herfindahl，越大越集中）。"""
    cum = p["amount_c"].groupby([p["instrument_id"], p["day"]]).cumsum()
    sh = p["amount_c"] / cum.where(cum > 0)
    return (sh ** 2).groupby([p["instrument_id"], p["day"]]).cumsum()


def n_iskew(p):
    """特质偏度（取负）：最近 40 窗口残差窗口收益的三阶矩 / 标准差^3。"""
    r = _log_ratio(p["px"], p.groupby("instrument_id", sort=False)["px"].shift(1))
    res = r - fl.market_beta(p) * r.groupby(p["date"]).transform("mean")
    m3 = grolling(res ** 3, p["instrument_id"], 40, "mean", 10)
    sd = grolling(res, p["instrument_id"], 40, "std", 10)
    return -m3 / (sd ** 3).where(sd > 0)


def n_maxloss(p):
    """最近 40 窗口最差单窗口收益（取负，尾部风险代理）。"""
    r = _log_ratio(p["px"], p.groupby("instrument_id", sort=False)["px"].shift(1))
    return -grolling(r, p["instrument_id"], 40, "min", 10)


def n_gap(p):
    """隔夜跳空反转向量：ln(昨收 / 当日开盘)。"""
    return _log_ratio(p["pre_close"], p["open"])


def n_amp(p):
    """近 20 窗口平均振幅（取负）。"""
    amp = (p["high"] - p["low"]) / p["pre_close"].where(p["pre_close"] > 0)
    return -grolling(amp, p["instrument_id"], 20, "mean", 5)


CANDS = {
    # R2 赢家的组合与权重扫描
    "104_ds0.10": blend(base_104, downside_share, 0.10),
    "104_ds0.15": blend(base_104, downside_share, 0.15),
    "104_ds0.20": blend(base_104, downside_share, 0.20),
    "104_d5_0.15": blend(base_104, d5_rev, 0.15),
    "104_d5_0.25": blend(base_104, d5_rev, 0.25),
    "104_cent0.15": blend(base_104, vol_centroid, 0.15),
    "104_ds0.15_d5_0.15": lambda p: rank_blend(p.assign(a=base_104(p), b=downside_share(p), c=d5_rev(p)),
                                               ["a", "b", "c"], [1.0, 0.15, 0.15]),
    "soft2_ds0.15": blend(base_soft2, downside_share, 0.15),
    "soft1_ds0.15": blend(base_soft1, downside_share, 0.15),
    # 新方向
    "104_herf": blend(base_104, n_herf, 0.15),
    "104_iskew": blend(base_104, n_iskew, 0.15),
    "104_maxloss": blend(base_104, n_maxloss, 0.15),
    "104_gap": blend(base_104, n_gap, 0.15),
    "104_amp": blend(base_104, n_amp, 0.15),
    "104_dvwap_s": blend(base_104, lambda q: q["r_dvwap_s"], 0.15),
}
