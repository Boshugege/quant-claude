# -*- coding: utf-8 -*-
"""R2 探索候选：同伴定义（软同伴/成交额加权）与中性化特征、力度，以及低换手附加项。

本地实现，只用过去数据。同伴相关性矩阵按月计算并缓存（键为面板对象 id）。
"""
import numpy as np
import pandas as pd

import factor_lib as fl
from factor_lib import _log_ratio, grolling, rank_blend, ladder, neutralize, neutral4, size_neg, ivol, amihud, max_effect

_WCACHE = {}


# ---------------------------------------------------------------- 同伴权重
def _daily_ret(p):
    gd = p.groupby(["instrument_id", "day"], sort=False)
    dr = _log_ratio(gd["px"].last(), gd["pre_close"].first()).unstack("instrument_id").sort_index()
    return dr


def peer_weights(p, k=100, power=1.0, min_obs=30, mode="soft"):
    """逐月计算同伴权重矩阵：mode=soft 用正相关^power（保留每行前 k 个后归一），mode=hard 用 top-k 等权。"""
    key = (id(p), k, power, mode, min_obs)
    if key in _WCACHE:
        return _WCACHE[key]
    dr = _daily_ret(p)
    insts = dr.columns.values
    days = dr.index
    months = days.to_period("M")
    out = {}
    for m in months.unique():
        first = days[months == m][0]
        prior = days[days < first][-60:]
        if len(prior) < min_obs:
            out[m] = None
            continue
        hist = dr[(days < first) & (days >= prior[0])]
        ok = (hist.notna().sum() >= min_obs).values
        sub = np.where(ok)[0]
        c = np.nan_to_num(hist.iloc[:, sub].corr(min_periods=min_obs).values, nan=0.0)
        np.fill_diagonal(c, 0.0)
        W = np.zeros((len(insts), len(insts)))
        if mode == "soft":
            w = np.clip(c, 0.0, None) ** power
            if k < w.shape[1]:
                thr = np.sort(w, axis=1)[:, -k][:, None]
                w = np.where(w >= thr, w, 0.0)
            W[np.ix_(sub, sub)] = w
            rs = W.sum(axis=1, keepdims=True)
            W = np.where(rs > 0, W / np.where(rs > 0, rs, 1.0), 0.0)
        else:
            idx = np.argsort(-c, axis=1)[:, :k]
            for r in range(len(sub)):
                W[sub[r], sub[idx[r]]] = 1.0
        out[m] = W
    _WCACHE[key] = (insts, out)
    return insts, out


def peer_mean_w(p, s, insts, W_by_month):
    """按同伴权重求加权均值（要求至少 3 个有效同伴，权重归一）。"""
    out = np.full(len(p), np.nan)
    months = p["day"].dt.to_period("M")
    for m, rows in p.groupby(months, sort=False).indices.items():
        W = W_by_month.get(m)
        if W is None:
            continue
        sub = p.iloc[rows]
        wide = pd.Series(np.asarray(s)[rows], index=pd.MultiIndex.from_arrays(
            [sub["date"].values, sub["instrument_id"].values])).unstack().reindex(columns=insts)
        X = wide.values
        M = (~np.isnan(X)).astype(np.float64)
        V = np.nan_to_num(X, nan=0.0)
        WT = W.T
        num = V @ WT
        den = M @ WT
        cnt = M @ (WT > 0).astype(np.float64)
        pm = np.where((den > 0) & (cnt >= 3), num / np.where(den > 0, den, 1.0), np.nan)
        pm = pd.DataFrame(pm, index=wide.index, columns=insts).stack(dropna=False)
        out[rows] = pm.reindex(pd.MultiIndex.from_arrays([sub["date"].values, sub["instrument_id"].values])).values
    return pd.Series(out, index=p.index)


def peer_res_ladder_w(p, parts=("30", "rv4", "rv8", "day"), mode="soft", power=1.0, k=100):
    insts, W = peer_weights(p, k=k, power=power, mode=mode)
    cols = {}
    for n in parts:
        x = {"30": p["r_30"], "day": p["r_day"]}.get(n) if n in ("30", "day") else fl.rv_raw(p, int(n[2:]))
        pm = peer_mean_w(p, x, insts, W)
        cols["c_" + n] = x - pm.fillna(x.groupby(p["date"]).transform("mean"))
    return rank_blend(p.assign(**cols), list(cols))


# ---------------------------------------------------------------- 附加特征
def c_turnover(p):
    """换手代理：最近 8 窗口成交额 / 最近 40 窗口成交额（相对放量程度）。"""
    a8 = grolling(p["amount_c"], p["instrument_id"], 8, "sum", 4)
    a40 = grolling(p["amount_c"], p["instrument_id"], 40, "sum", 10)
    return np.log((a8 / a40.where(a40 > 0)).where(a40 > 0))


def c_px_sq(p):
    """价格水平非线性项：-(ln 价格)^2 的截面排名（极端低价/高价的非线性暴露）。"""
    return -(np.log(p["px"].where(p["px"] > 0)) ** 2)


def c_gap(p):
    """隔夜跳空：ln(开盘 / 昨收)。"""
    return _log_ratio(p["open"], p["pre_close"])


def c_amp(p):
    """近 20 窗口平均振幅（(高-低)/昨收）。"""
    amp = (p["high"] - p["low"]) / p["pre_close"].where(p["pre_close"] > 0)
    return grolling(amp, p["instrument_id"], 20, "mean", 5)


# ---------------------------------------------------------------- 候选
def r2_104_frac(p, frac):
    return neutral4(p, fl.peer_res_ladder(p), frac)


def r2_soft_peer_1(p):
    return peer_res_ladder_w(p, mode="soft", power=1.0, k=100)


def r2_soft_peer_2(p):
    return peer_res_ladder_w(p, mode="soft", power=2.0, k=100)


def r2_hard_peer_60(p):
    return peer_res_ladder_w(p, mode="hard", k=60)


def r2_soft_peer_1_neutral(p):
    return neutral4(p, r2_soft_peer_1(p), 1.5)


def r2_soft_peer_2_neutral(p):
    return neutral4(p, r2_soft_peer_2(p), 1.5)


def r2_day_in_peer(p):
    """把 20 日尺度反转作为第五个分量放进同伴残差阶梯，再做 1.5 倍中性化。"""
    return neutral4(p, peer_res_ladder_w(p, parts=("30", "rv4", "rv8", "day"), mode="soft", power=1.0, k=100), 1.5)


def r2_104_plus_max(p):
    return neutralize(p, fl.peer_res_ladder(p), [size_neg(p), ivol(p), amihud(p), -np.log(p["px"]), max_effect(p)], 1.5)


def r2_104_plus_turn(p):
    return neutralize(p, fl.peer_res_ladder(p), [size_neg(p), ivol(p), amihud(p), -np.log(p["px"]), c_turnover(p)], 1.5)


def r2_104_plus_px2(p):
    return neutralize(p, fl.peer_res_ladder(p), [size_neg(p), ivol(p), amihud(p), -np.log(p["px"]), c_px_sq(p)], 1.5)


def _add_low_turnover(p, col, w):
    base = neutral4(p, fl.peer_res_ladder(p), 1.5)
    return rank_blend(p.assign(a=base, b=col(p)), ["a", "b"], [1.0, w])


def r2_104_w_downside(p):
    return _add_low_turnover(p, lambda q: _downside(q), 0.15)


def r2_104_w_d5(p):
    from cand_r1 import d5_rev
    return _add_low_turnover(p, d5_rev, 0.15)


def _downside(q):
    r = _log_ratio(q["px"], q.groupby("instrument_id", sort=False)["px"].shift(1))
    sn = grolling(r.where(r < 0, 0.0) ** 2, q["instrument_id"], 40, "sum", 8)
    st = grolling(r ** 2, q["instrument_id"], 40, "sum", 8)
    return sn / st.where(st > 0)


CANDS = {
    "104_frac2.0": lambda p: r2_104_frac(p, 2.0),
    "104_frac2.5": lambda p: r2_104_frac(p, 2.5),
    "104_frac3.0": lambda p: r2_104_frac(p, 3.0),
    "soft_peer_p1": r2_soft_peer_1,
    "soft_peer_p2": r2_soft_peer_2,
    "hard_peer_k60": r2_hard_peer_60,
    "soft_peer_p1_neutral": r2_soft_peer_1_neutral,
    "soft_peer_p2_neutral": r2_soft_peer_2_neutral,
    "104_plus_max": r2_104_plus_max,
    "104_plus_turn": r2_104_plus_turn,
    "104_plus_px2": r2_104_plus_px2,
    "104_w_downside": r2_104_w_downside,
    "104_w_d5": r2_104_w_d5,
}
