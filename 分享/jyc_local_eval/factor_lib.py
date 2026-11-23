# -*- coding: utf-8 -*-
"""因子库（示例）：用 30 分钟 bar 在本地复刻平台提交的因子（pandas 实现），并登记已知公榜分数。

用法：
- 在 FACTORS 中按 (编号, 名称, 公榜分数或 None, 计算函数, 备注) 登记因子；
- 计算函数接收面板 p（每行 = 一只股票的一根 30 分钟 bar，已按 股票、时间 排序），返回与 p 行对齐的 Series；
  因子值越大表示预期未来收益越高；缺失会按 0 处理（与平台提交代码“最终 fillna(0)”一致）；
- 公榜分数填你自己在平台上的得分，填得越多，run.py 输出的“本地综合分 vs 公榜”校准越可靠（至少 3 个才计算）。

只允许使用当前及过去的数据（shift(1)、滚动窗口等）；标签列（y_*、next_vwap）严禁在因子中使用。
"""
import numpy as np
import pandas as pd


def _log_ratio(num, den):
    num = pd.Series(num)
    den = pd.Series(den)
    ok = (num > 0) & (den > 0)
    return np.log(num.where(ok) / den.where(ok))


def grolling(s, keys, window, fn="mean", min_periods=1):
    """分组滚动（向量化）：等价于 s.groupby(keys).transform(lambda x: x.rolling(window).fn())，但快得多。"""
    r = getattr(s.groupby(keys, sort=False).rolling(window, min_periods=min_periods), fn)()
    r = r.reset_index(level=list(range(r.index.nlevels - 1)), drop=True)
    return r.reindex(s.index)


def add_components(p):
    """在面板上一次性计算示例因子共用的中间量（只用过去数据）。"""
    g = p.groupby("instrument_id", sort=False)
    gd = p.groupby(["instrument_id", "day"], sort=False)
    same_day_prev = g["day"].shift(1) == p["day"]

    cum_amt = gd["amount_c"].cumsum()
    cum_vol = gd["volume_c"].cumsum()
    p["day_vwap"] = (cum_amt / cum_vol.where(cum_vol > 0)).where(cum_amt > 0)
    p["prev_px"] = g["px"].shift(1).where(same_day_prev, p["pre_close"])

    win_ret = _log_ratio(p["px"], g["px"].shift(1))
    sigma = grolling(win_ret, p["instrument_id"], 40, "std", min_periods=8)
    sigma = sigma.where(sigma > 0)
    p["sigma"] = sigma.fillna(sigma.groupby(p["date"]).transform("median"))

    p["r_day"] = _log_ratio(p["pre_close"], p["px"])        # 日内反转：ln(昨收 / 当前价)
    p["r_30"] = _log_ratio(p["prev_px"], p["px"])           # 30 分钟反转
    p["r_vw"] = _log_ratio(p["vwap_w"], p["px"])            # 本窗口 VWAP 偏离
    p["r_dvwap"] = _log_ratio(p["day_vwap"], p["px"])       # 日内累计 VWAP 偏离

    hi = p["high"].where(p["high"] > 0).groupby([p["instrument_id"], p["day"]]).cummax()
    lo = p["low"].where(p["low"] > 0).groupby([p["instrument_id"], p["day"]]).cummin()
    H = np.fmax(hi, p["pre_close"])
    L = np.fmin(lo, p["pre_close"])
    rng = (H - L).where((H - L) > 0)
    p["r_rng"] = (H - p["px"]) / rng - 0.5                  # 日内区间位置（越靠近低点越大）

    # 委托笔数不平衡（bar 末快照、3 档；平台上是逐分钟 5 档，这里只是近似）。
    sb = sum(p[f"bid_num_orders{i}"] for i in (1, 2, 3))
    sa = sum(p[f"ask_num_orders{i}"] for i in (1, 2, 3))
    p["cnt_sum"] = (sb - sa) / (sb + sa + 1e-8)
    return p


def rank_blend(p, cols, weights=None):
    """各分量在每个截面内居中排名后加权平均（缺失分量不参与）。"""
    weights = weights or [1.0] * len(cols)
    r = p[cols].groupby(p["date"]).rank(pct=True) - 0.5
    w = np.array(weights, dtype=float)
    vals = r.to_numpy()
    has = ~np.isnan(vals)
    den = (has * w).sum(axis=1)
    num = np.nansum(np.nan_to_num(vals) * w, axis=1)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=p.index)


# (编号, 名称, 已知公榜分数或 None, 计算函数, 备注)
# 公榜分数请替换为你自己的平台得分（None 表示未知，不参与校准）。
FACTORS = [
    ("E1", "30分钟反转", None, lambda p: p["r_30"], ""),
    ("E2", "日内反转", None, lambda p: p["r_day"], ""),
    ("E3", "VWAP偏离反转", None, lambda p: p["r_vw"], ""),
    ("E4", "日内VWAP偏离反转", None, lambda p: p["r_dvwap"], ""),
    ("E5", "波动标准化日内VWAP反转", None, lambda p: (p["r_dvwap"] / p["sigma"]).fillna(p["r_dvwap"]), ""),
    ("E6", "日内区间位置反转", None, lambda p: p["r_rng"], ""),
    ("E7", "日内+30分钟反转", None, lambda p: rank_blend(p, ["r_day", "r_30"]), ""),
    ("E8", "30分钟动量(反向示例)", None, lambda p: -p["r_30"], "方向相反，应得低分"),
    ("E9", "委托笔数不平衡(近似)", None, lambda p: p["cnt_sum"], "approx"),
]
