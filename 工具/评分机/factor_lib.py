# -*- coding: utf-8 -*-
"""因子库：用 30 分钟 bar 在本地复刻平台提交的因子（pandas 实现），并登记已知公榜分数。

说明：
- 价格类因子（反转家族）只依赖窗口级数据，本地复刻与云端提交代码口径一致；
- 盘口/委托笔数类因子云端是“30 分钟内逐分钟均值、5 档”，本地只有 bar 末快照、3 档，属于近似（标注 approx）。
"""
import numpy as np
import pandas as pd


def _log_ratio(num, den):
    num = pd.Series(num)
    den = pd.Series(den)
    ok = (num > 0) & (den > 0)
    return np.log(num.where(ok) / den.where(ok))


def grolling(s, keys, window, fn="mean", min_periods=1):
    """分组滚动（向量化，C 实现）：等价于 s.groupby(keys).transform(lambda x: x.rolling(window).fn())，但快得多。"""
    r = getattr(s.groupby(keys, sort=False).rolling(window, min_periods=min_periods), fn)()
    r = r.reset_index(level=list(range(r.index.nlevels - 1)), drop=True)
    return r.reindex(s.index)


def add_components(p):
    """在面板上一次性计算各因子共用的中间量（只用过去数据）。"""
    g = p.groupby("instrument_id", sort=False)
    gd = p.groupby(["instrument_id", "day"], sort=False)
    same_day_prev = g["day"].shift(1) == p["day"]

    cum_amt = gd["amount_c"].cumsum()
    cum_vol = gd["volume_c"].cumsum()
    p["day_vwap"] = (cum_amt / cum_vol.where(cum_vol > 0)).where(cum_amt > 0)
    roll_amt = grolling(p["amount_c"], p["instrument_id"], 8, "sum")
    roll_vol = grolling(p["volume_c"], p["instrument_id"], 8, "sum")
    p["roll_vwap"] = (roll_amt / roll_vol.where(roll_vol > 0)).where(roll_amt > 0)
    p["prev_px"] = g["px"].shift(1).where(same_day_prev, p["pre_close"])

    win_ret = _log_ratio(p["px"], g["px"].shift(1))
    sigma = grolling(win_ret, p["instrument_id"], 40, "std", min_periods=8)
    sigma = sigma.where(sigma > 0)
    p["sigma"] = sigma.fillna(sigma.groupby(p["date"]).transform("median"))

    p["r_day"] = _log_ratio(p["pre_close"], p["px"])
    p["r_30"] = _log_ratio(p["prev_px"], p["px"])
    p["r_vw"] = _log_ratio(p["vwap_w"], p["px"])
    p["r_dvwap"] = _log_ratio(p["day_vwap"], p["px"])
    p["r_roll"] = _log_ratio(p["roll_vwap"], p["px"]).fillna(p["r_dvwap"])
    p["r_rv"] = (p["r_roll"] / p["sigma"]).fillna(p["r_roll"])
    p["r_dvwap_s"] = (p["r_dvwap"] / p["sigma"]).fillna(p["r_dvwap"])

    hi = p["high"].where(p["high"] > 0).groupby([p["instrument_id"], p["day"]]).cummax()
    lo = p["low"].where(p["low"] > 0).groupby([p["instrument_id"], p["day"]]).cummin()
    H = np.fmax(hi, p["pre_close"])
    L = np.fmin(lo, p["pre_close"])
    rng = (H - L).where((H - L) > 0)
    p["r_rng"] = (H - p["px"]) / rng - 0.5
    p["r_dvwap_rng"] = (p["day_vwap"] - p["px"]) / rng

    day_open = gd["open"].transform("first")
    p["r_open"] = _log_ratio(day_open, p["px"])
    p["r_night"] = _log_ratio(p["pre_close"], day_open)
    p["adj_px"] = p["px"] * p["adjust_factor"]

    # 委托笔数（bar 末快照、3 档，近似云端的 5 档逐分钟均值）。
    w = {1: np.exp(-0.6), 2: np.exp(-0.3), 3: 1.0}
    fb = sum(p[f"bid_num_orders{i}"] * w[i] for i in (1, 2, 3))
    fa = sum(p[f"ask_num_orders{i}"] * w[i] for i in (1, 2, 3))
    p["cnt_far"] = (fb - fa) / (fb + fa + 1e-8)
    sb = sum(p[f"bid_num_orders{i}"] for i in (1, 2, 3))
    sa = sum(p[f"ask_num_orders{i}"] for i in (1, 2, 3))
    p["cnt_sum"] = (sb - sa) / (sb + sa + 1e-8)

    # 同时段历史：按（股票，时段）分组，先后移一天再滚动（只用过去数据）。
    def past_same_slot(col, n, min_n):
        keys = [p["instrument_id"], p["slot"]]
        shifted = p[col].groupby(keys, sort=False).shift(1)
        return grolling(shifted, keys, n, "mean", min_periods=min_n)
    p["slot_ret"] = _log_ratio(p["next_vwap"], p["px"])
    p["r_slot"] = past_same_slot("slot_ret", 20, 5)
    p["cum_vol_today"] = cum_vol
    p["w_relvol"] = cum_vol / past_same_slot("cum_vol_today", 10, 3)
    cum_deals = gd["deal_number"].cumsum()
    p["avg_trade"] = (cum_amt / cum_deals.where(cum_deals > 0)).where(cum_amt > 0)
    p["w_relsize"] = p["avg_trade"] / past_same_slot("avg_trade", 10, 3)
    p["r_5d"] = _log_ratio(p.groupby(["instrument_id", "slot"], sort=False)["adj_px"].shift(5), p["adj_px"]).fillna(p["r_day"])
    return p


def rank_blend(p, cols, weights=None):
    """各分量在每个截面内居中排名后加权平均（缺失分量不参与），与云端复合因子一致。"""
    weights = weights or [1.0] * len(cols)
    r = p[cols].groupby(p["date"]).rank(pct=True) - 0.5
    w = np.array(weights, dtype=float)
    vals = r.to_numpy()
    has = ~np.isnan(vals)
    den = (has * w).sum(axis=1)
    num = np.nansum(np.nan_to_num(vals) * w, axis=1)
    return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=p.index)


def rv_raw(p, k):
    """滚动 k 个 30 分钟窗口（跨日）的清洗后成交 VWAP 偏离 ln(滚动 VWAP / 当前价)，缺失时退回日内累计 VWAP。"""
    col = f"_rv{k}"
    if col in p.columns:
        return p[col]
    a = grolling(p["amount_c"], p["instrument_id"], k, "sum")
    v = grolling(p["volume_c"], p["instrument_id"], k, "sum")
    p[col] = _log_ratio((a / v.where(v > 0)).where(a > 0), p["px"]).fillna(p["r_dvwap"])
    return p[col]


def ladder(p, parts, normalize=False):
    """多尺度反转阶梯：parts 取自 {"30", "vw", "day", "rv<k>"}，可选各分量除以 sigma，截面排名等权。"""
    def comp(n):
        if n in ("30", "vw", "day"):
            return p[{"30": "r_30", "vw": "r_vw", "day": "r_day"}[n]]
        return rv_raw(p, int(n[2:]))
    cols = {}
    for n in parts:
        s = comp(n)
        cols["c_" + n] = (s / p["sigma"]).fillna(s) if normalize else s
    return rank_blend(p.assign(**cols), list(cols))


def market_beta(p):
    """大盘（截面等权）窗口收益与个股滚动 beta（最近 40 个窗口，至少 10 个，截断 [-1, 4]，不足取 1）。"""
    if "_beta" not in p.columns:
        g = p.groupby("instrument_id", sort=False)
        same = g["day"].shift(1) == p["day"]
        ret30 = np.log(p["px"] / g["px"].shift(1)).where(same).fillna(np.log(p["px"] / p["pre_close"]))
        mkt30 = ret30.groupby(p["date"]).transform("mean")
        rm = lambda s: grolling(s, p["instrument_id"], 40, "mean", min_periods=10)
        ex, ey, exy, ey2 = rm(ret30), rm(mkt30), rm(ret30 * mkt30), rm(mkt30 ** 2)
        var_m = (ey2 - ey ** 2).where((ey2 - ey ** 2) > 1e-12)
        p["_beta"] = ((exy - ex * ey) / var_m).clip(-1, 4).fillna(1.0)
    return p["_beta"]


def residual(p, s):
    """分量 s 去掉 beta x 大盘同口径分量（截面均值），只保留个股自身的偏离。"""
    return s - market_beta(p) * s.groupby(p["date"]).transform("mean")


def res_ladder(p, parts):
    """残差版多尺度阶梯：各分量残差化后截面排名等权。"""
    def comp(n):
        if n in ("30", "vw", "day"):
            return p[{"30": "r_30", "vw": "r_vw", "day": "r_day"}[n]]
        return rv_raw(p, int(n[2:]))
    cols = {"c_" + n: residual(p, comp(n)) for n in parts}
    return rank_blend(p.assign(**cols), list(cols))


def amihud(p, k=40):
    """非流动性：最近 k 个窗口“|窗口收益| / 窗口成交额（亿元）”的均值（至少 10 个窗口）。"""
    g = p.groupby("instrument_id", sort=False)
    il = np.log(p["px"] / g["px"].shift(1)).abs() / (p["amount_c"] / 1e8).where(p["amount_c"] > 0)
    return grolling(il, p["instrument_id"], k, "mean", min_periods=10)


def _win_ret(p):
    return np.log(p["px"] / p.groupby("instrument_id", sort=False)["px"].shift(1))


def ivol(p):
    """特质波动（取负）：最近 40 个窗口残差窗口收益（窗口收益 - beta x 截面均值）的标准差。"""
    r = _win_ret(p)
    res = r - market_beta(p) * r.groupby(p["date"]).transform("mean")
    return -grolling(res, p["instrument_id"], 40, "std", min_periods=10)


def max_effect(p):
    """MAX（取负）：最近 40 个窗口中最大的窗口收益（彩票偏好）。"""
    return -grolling(_win_ret(p), p["instrument_id"], 40, "max", min_periods=10)


def char_blend(p, base, extras, weights):
    """主信号（权重 1）与若干特征分量做截面排名加权平均。"""
    cols = {"a": base, **{f"e{i}": e for i, e in enumerate(extras)}}
    return rank_blend(p.assign(**cols), list(cols), [1.0] + list(weights))


def ladder93(p, parts):
    """93 号结构：原始阶梯与残差阶梯合成。"""
    return rank_blend(p.assign(_a=ladder(p, parts), _b=res_ladder(p, parts)), ["_a", "_b"])


def ladder_by_slot(p):
    """按时段调整：14:00 截面只用短周期分量（30 分钟、1 小时、2 小时），其余时段用五尺度。"""
    full = ladder93(p, ["30", "rv2", "rv4", "rv8", "day"])
    short = ladder93(p, ["30", "rv2", "rv4"])
    return full.where(p["slot"] != 1400, short)


def rank_weighted(p, base, weight_col, invert):
    z = p[base].groupby(p["date"]).rank(pct=True) - 0.5
    w = p[weight_col].groupby(p["date"]).rank(pct=True).fillna(0.5)
    return z * ((1.5 - w) if invert else (0.5 + w))


def size_neg(p):
    """成交额规模（取负，小盘代理）：-ln(1 + 最近 40 个窗口清洗后成交额之和)。"""
    return -np.log1p(grolling(p["amount_c"], p["instrument_id"], 40, "sum", min_periods=10))


def neutralize(p, s, chars, frac=1.0):
    """截面中性化：因子居中排名对各特征居中排名（缺失记 0）做逐截面 OLS（含截距），返回 排名 - frac x 拟合值（不扣截距）。"""
    y = s.groupby(p["date"]).rank(pct=True) - 0.5
    X = np.column_stack([(c.groupby(p["date"]).rank(pct=True) - 0.5).fillna(0.0).values for c in chars])
    ok = y.notna().values
    codes, uniq = pd.factorize(p["date"].values)
    k, G = X.shape[1], len(uniq)
    Xc = np.column_stack([np.ones(len(p)), X]) * ok[:, None]
    yv = np.nan_to_num(y.values) * ok
    XtX = np.zeros((G, k + 1, k + 1))
    Xty = np.zeros((G, k + 1))
    for i in range(k + 1):
        Xty[:, i] = np.bincount(codes, Xc[:, i] * yv, minlength=G)
        for j in range(i, k + 1):
            v = np.bincount(codes, Xc[:, i] * Xc[:, j], minlength=G)
            XtX[:, i, j] = v
            XtX[:, j, i] = v
    b = np.linalg.solve(XtX + 1e-9 * np.eye(k + 1), Xty[..., None])[..., 0]
    fit = (X * b[codes, 1:]).sum(axis=1)
    return pd.Series(np.where(ok, y.values - frac * fit, np.nan), index=p.index)


def neutral4(p, s, frac=1.0):
    """对四个股票特征（规模、特质波动、非流动性、低价）中性化。"""
    return neutralize(p, s, [size_neg(p), ivol(p), amihud(p), -np.log(p["px"])], frac)


_PEERS = {}


def peer_index(p, k=30, lookback=60, min_obs=30, freq="day"):
    """同伴（伪行业）：每个自然月月初，用此前 lookback 个交易日的日收益 ln(收盘段价格 / 昨收) 计算相关系数，
    每只股票取相关性最高的 k 只（至少 min_obs 个共同观测）。返回 {月份: (股票代码数组, 同伴下标矩阵或 None)}。"""
    key = (id(p), k, lookback, freq)
    if key in _PEERS:
        return _PEERS[key]
    if freq == "day":
        gd = p.groupby(["instrument_id", "day"], sort=False)
        dr = _log_ratio(gd["px"].last(), gd["pre_close"].first()).unstack("instrument_id").sort_index()
        row_day = dr.index
    else:   # 30 分钟收益（日内首个窗口相对昨收），回看 lookback 个交易日内的全部窗口
        prev = p.groupby("instrument_id", sort=False)["px"].shift(1).where(
            p.groupby("instrument_id", sort=False)["day"].shift(1) == p["day"], p["pre_close"])
        r = _log_ratio(p["px"], prev)
        dr = pd.Series(r.values, index=pd.MultiIndex.from_arrays([p["date"].values, p["instrument_id"].values])).unstack().sort_index()
        row_day = dr.index.normalize()
    insts = dr.columns.values
    days = pd.DatetimeIndex(sorted(set(row_day)))
    months = days.to_period("M")
    out = {}
    for m in months.unique():
        first = days[months == m][0]
        prior_days = days[days < first][-lookback:]
        hist = dr[(row_day < first) & (row_day >= (prior_days[0] if len(prior_days) else first))]
        if len(prior_days) < min_obs:
            out[m] = (insts, None)
            continue
        need = min_obs if freq == "day" else min_obs * 4
        ok = (hist.notna().sum() >= need).values
        c = hist.loc[:, ok].corr(min_periods=need).values
        np.fill_diagonal(c, -np.inf)
        c = np.nan_to_num(c, nan=-np.inf)
        idx = np.argsort(-c, axis=1, kind="stable")[:, :k]
        sub = np.where(ok)[0]
        full = np.full((len(insts), k), -1, dtype=np.int64)
        full[sub] = sub[idx]
        out[m] = (insts, full)
    _PEERS[key] = out
    return out


def peer_mean(p, s, k=30, freq="day"):
    """同伴均值：每个截面上该股 k 个同伴的 s 均值（至少 3 个有效值），无同伴时为缺失。"""
    peers = peer_index(p, k, freq=freq)
    out = np.full(len(p), np.nan)
    months = p["day"].dt.to_period("M")
    for m, rows in p.groupby(months, sort=False).indices.items():
        insts, idx = peers.get(m, (None, None))
        if idx is None:
            continue
        sub = p.iloc[rows]
        wide = pd.Series(s.values[rows], index=pd.MultiIndex.from_arrays([sub["date"].values, sub["instrument_id"].values])).unstack()
        wide = wide.reindex(columns=insts)
        X = wide.values
        valid = idx >= 0
        G = X[:, np.where(valid, idx, 0)]
        G = np.where(valid[None, :, :], G, np.nan)
        cnt = (~np.isnan(G)).sum(axis=2)
        pm = np.where(cnt >= 3, np.nansum(G, axis=2) / np.maximum(cnt, 1), np.nan)
        pm = pd.DataFrame(pm, index=wide.index, columns=insts).stack(dropna=False)
        out[rows] = pm.reindex(pd.MultiIndex.from_arrays([sub["date"].values, sub["instrument_id"].values])).values
    return pd.Series(out, index=p.index)


def peer_res_ladder(p, parts=("30", "rv4", "rv8", "day"), k=30, freq="day"):
    """同伴残差阶梯：各反转分量减去同伴均值（无同伴时减截面均值），再截面排名等权。"""
    comp = {"30": p["r_30"], "day": p["r_day"]}
    cols = {}
    for n in parts:
        x = comp[n] if n in comp else rv_raw(p, int(n[2:]))
        cols["c_" + n] = x - peer_mean(p, x, k, freq).fillna(x.groupby(p["date"]).transform("mean"))
    return rank_blend(p.assign(**cols), list(cols))


# (编号, 名称, 已知公榜分数或 None, 计算函数, 备注)
FACTORS = [
    ("24", "委托笔数不平衡(近似)", 0.40672, lambda p: p["cnt_sum"], "approx"),
    ("41", "委托笔数远档加权(近似)", 0.41587, lambda p: p["cnt_far"], "approx"),
    ("39", "VWAP偏离反转", 0.40763, lambda p: p["r_vw"], ""),
    ("44", "笔数+VWAP动量(方向错)", 0.05761, lambda p: rank_blend(p.assign(m=-p["r_vw"]), ["cnt_far", "m"]), "approx"),
    ("47", "笔数+VWAP反转", 0.46613, lambda p: rank_blend(p, ["cnt_far", "r_vw"]), "approx"),
    ("49", "30分钟动量(方向错)", 0.02840, lambda p: -p["r_30"], ""),
    ("50", "日内动量(方向错)", 0.20203, lambda p: -p["r_day"], ""),
    ("51", "VWAP窗口间动量(方向错)", 0.04868,
     lambda p: _log_ratio(p["vwap_w"], p.groupby("instrument_id")["vwap_w"].shift(1).where(
         p.groupby("instrument_id")["day"].shift(1) == p["day"], p["pre_close"])), ""),
    ("56", "日内反转", 0.60243, lambda p: p["r_day"], ""),
    ("57", "30分钟反转", 0.57571, lambda p: p["r_30"], ""),
    ("59", "买卖价差反弹", 0.08947, lambda p: _log_ratio(p["px"], p["close"]), ""),
    ("60", "五日同时刻反转", 0.39113, lambda p: p["r_5d"], ""),
    ("61", "日内区间位置反转", 0.65968, lambda p: p["r_rng"], ""),
    ("63", "开盘后日内反转", 0.57944, lambda p: p["r_open"], ""),
    ("64", "日内+30分钟反转", 0.77358, lambda p: rank_blend(p, ["r_day", "r_30"]), ""),
    ("65", "双尺度反转+笔数", 0.58089, lambda p: rank_blend(p, ["r_day", "r_30", "cnt_far"]), "approx"),
    ("66", "日内反转+笔数", 0.55732, lambda p: rank_blend(p, ["r_day", "cnt_far"]), "approx"),
    ("67", "日内VWAP偏离反转", 0.73757, lambda p: p["r_dvwap"], ""),
    ("68", "日内VWAP+30分钟", 0.68092, lambda p: rank_blend(p, ["r_dvwap", "r_30"]), ""),
    ("69", "三尺度反转", 0.78420, lambda p: rank_blend(p, ["r_dvwap", "r_day", "r_30"]), ""),
    ("71", "滚动240分钟VWAP反转", 0.78304, lambda p: p["r_roll"], ""),
    ("73", "波动标准化日内VWAP反转", 0.78343, lambda p: p["r_dvwap_s"], ""),
    ("74", "振幅标准化日内VWAP反转", 0.73565, lambda p: p["r_dvwap_rng"], ""),
    ("75", "三基准反转", 0.74721, lambda p: rank_blend(p, ["r_dvwap", "r_rng", "r_30"]), ""),
    ("76", "振幅标准化VWAP+30分钟", 0.67861, lambda p: rank_blend(p, ["r_dvwap_rng", "r_30"]), ""),
    ("80", "滚动VWAP波动标准化反转", 0.77935, lambda p: p["r_rv"], ""),
    ("81", "滚动VWAP标准化三尺度", 0.78967, lambda p: rank_blend(p, ["r_rv", "r_day", "r_30"]), ""),
    ("82", "滚动VWAP标准化+日内反转", 0.7044, lambda p: rank_blend(p, ["r_rv", "r_day"]), ""),
    ("83", "全标准化三尺度", 0.77706,
     lambda p: rank_blend(p.assign(a=p["r_day"] / p["sigma"], b=p["r_30"] / p["sigma"]), ["r_rv", "a", "b"]), ""),
    ("84", "同时段收益延续", 0.04015, lambda p: p["r_slot"], ""),
    ("85", "放量加权反转", 0.74226, lambda p: rank_weighted(p, "r_rv", "w_relvol", invert=False), ""),
    ("86", "小单主导加权反转", 0.72122, lambda p: rank_weighted(p, "r_rv", "w_relsize", invert=True), ""),
    ("87", "隔夜反转", 0.2696, lambda p: p["r_night"], ""),
    # 第十一批：本地评分筛选（开发/验证/保留三段均优于 64 号）
    ("88", "多尺度阶梯双版本合成", 0.80759,
     lambda p: rank_blend(p.assign(a=ladder(p, ["30", "rv4", "rv8", "day"], False),
                                   b=ladder(p, ["30", "rv4", "rv8", "day"], True)), ["a", "b"]), ""),
    ("89", "两小时VWAP反转", 0.76509, lambda p: rv_raw(p, 4), ""),
    ("90", "多尺度反转阶梯", 0.81328, lambda p: ladder(p, ["30", "rv4", "rv8", "day"], False), ""),
    ("91", "五尺度反转阶梯", 0.76964, lambda p: ladder(p, ["30", "vw", "rv4", "rv8", "day"], False), ""),
    # 第十二批：五尺度阶梯（加 1 小时）与残差反转合成
    ("92", "五尺度VWAP反转阶梯", 0.77274, lambda p: ladder(p, ["30", "rv2", "rv4", "rv8", "day"], False), ""),
    ("93", "五尺度阶梯+残差反转合成", 0.77044,
     lambda p: rank_blend(p.assign(a=ladder(p, ["30", "rv2", "rv4", "rv8", "day"], False),
                                   b=res_ladder(p, ["30", "rv2", "rv4", "rv8", "day"])), ["a", "b"]), ""),
    # 第十三批：按时段阶梯与股票特征分量（非流动性、低价）
    ("94", "按时段阶梯+非流动性+低价", 0.52621,
     lambda p: rank_blend(p.assign(a=ladder_by_slot(p), m=amihud(p), n=-np.log(p["px"])), ["a", "m", "n"], [1.0, 0.2, 0.2]), ""),
    ("95", "按时段阶梯+非流动性", 0.51456,
     lambda p: rank_blend(p.assign(a=ladder_by_slot(p), m=amihud(p)), ["a", "m"], [1.0, 0.2]), ""),
    ("96", "五尺度残差阶梯+低价", 0.78058,
     lambda p: rank_blend(p.assign(a=ladder93(p, ["30", "rv2", "rv4", "rv8", "day"]), n=-np.log(p["px"])), ["a", "n"], [1.0, 0.2]), ""),
    ("97", "按时段五尺度残差阶梯", 0.74097, lambda p: ladder_by_slot(p), ""),
    # 第十四批：90 号核心 + 股票特征（非流动性、低价、特质波动、MAX）
    ("98", "90+四特征", 0.73515,
     lambda p: char_blend(p, ladder(p, ["30", "rv4", "rv8", "day"]), [amihud(p), -np.log(p["px"]), ivol(p), max_effect(p)], [0.2] * 4), ""),
    ("99", "90+三特征", 0.69553,
     lambda p: char_blend(p, ladder(p, ["30", "rv4", "rv8", "day"]), [amihud(p), -np.log(p["px"]), ivol(p)], [0.2] * 3), ""),
    ("100", "90+特质波动", 0.68854,
     lambda p: char_blend(p, ladder(p, ["30", "rv4", "rv8", "day"]), [ivol(p)], [0.2]), ""),
    ("101", "88+三特征", 0.70485,
     lambda p: char_blend(p, rank_blend(p.assign(a=ladder(p, ["30", "rv4", "rv8", "day"]), b=ladder(p, ["30", "rv4", "rv8", "day"], True)), ["a", "b"]),
                          [amihud(p), -np.log(p["px"]), ivol(p)], [0.2] * 3), ""),
    ("102", "纯股票特征", 0.3701,
     lambda p: rank_blend(p.assign(a=amihud(p), b=-np.log(p["px"]), c=ivol(p), d=max_effect(p)), ["a", "b", "c", "d"]), ""),
    # 第十五批：同伴（伪行业）残差反转 + 四特征截面中性化
    ("103", "同伴残差阶梯四特征中性", None, lambda p: neutral4(p, peer_res_ladder(p)), ""),
    ("104", "同伴残差阶梯过度中性", 0.83716, lambda p: neutral4(p, peer_res_ladder(p), 1.5), ""),
    ("105", "四尺度阶梯四特征中性", 0.83372, lambda p: neutral4(p, ladder(p, ["30", "rv4", "rv8", "day"])), ""),
    ("106", "同伴中性阶梯加四特征", 0.74751,
     lambda p: char_blend(p, neutral4(p, peer_res_ladder(p)), [amihud(p), -np.log(p["px"]), ivol(p), max_effect(p)], [0.1] * 4), ""),
    ("107", "同伴残差反转阶梯", 0.82184, lambda p: peer_res_ladder(p), ""),
    # 第十六批：沿“中性化力度 / 特征选择 / 同伴定义”三条线推进 104 号
    ("108", "同伴残差阶梯二倍中性", None, lambda p: neutral4(p, peer_res_ladder(p), 2.0), ""),
    ("109", "同伴残差阶梯二点五倍中性", None, lambda p: neutral4(p, peer_res_ladder(p), 2.5), ""),
    ("110", "同伴残差阶梯规模波动二倍中性", None, lambda p: neutralize(p, peer_res_ladder(p), [size_neg(p), ivol(p)], 2.0), ""),
    ("111", "分钟同伴残差阶梯二倍中性", None, lambda p: neutral4(p, peer_res_ladder(p, freq="30m"), 2.0), ""),
    ("112", "双阶梯合成四特征中性", None,
     lambda p: neutral4(p, rank_blend(p.assign(a=ladder(p, ["30", "rv4", "rv8", "day"]), b=peer_res_ladder(p)), ["a", "b"]), 1.5), ""),
]
