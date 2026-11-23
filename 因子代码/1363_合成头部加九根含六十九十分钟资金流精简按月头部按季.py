def main(datasources, start_date, end_date):
    """【新根因子 第三轮·多根合成 1363】合成头部加九根含六十九十分钟资金流精简按月头部按季

    B = 490 头部原始值 + Σ w·rank(根)，09:30 置 0，股票池骨架、缺失填 0。根来自多统计量框架 1340_资金流窗口框架.py（原样复用），
    以 L_ 开头的季节性根先在同一股票连续截面上做 EWMA（α=0.75）再排名。各根含义：
      L_lab10  同时段季节性（下一个半小时段过去 10 日"收盘 ÷ VWAP − 1"均值）
      M_q_lin  聪明钱（|r|/v 加权的当日 VWAP ÷ 当日 VWAP − 1）
      N_dir    大单方向（Σ r·(ln 单笔额 − 之前分钟均值) ÷ Σ|r|）
      W_vdir   量价同向（Σ r·(ln 成交量 − 之前分钟均值) ÷ Σ|r|）
      W_zero   当日零收益分钟占比
      Y_sd5    近 5 日单笔成交额离散度
      X_n5d    近 5 日大单方向
      Z_mfr60  近 60 分钟上涨分钟成交量占比（资金流）
      Z_mfr90  近 90 分钟上涨分钟成交量占比（资金流）
    权重：{"L_lab10": +0.20, "M_q_lin": +0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": +0.10, "Y_sd5": +0.10, "X_n5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}。口径与评测机 values 合成（build_values_composite.py raw 模式）一致。
    """
    import numpy as np
    import pandas as pd
    import dai

    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": +0.10, "Y_sd5": +0.10, "X_n5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}
    ALPHA = 0.75
    MONTHLY = True                                                       # True：分钟类根按月分批（控制平台整段调用的内存）
    HEADQ = True                                                           # True：头部按自然季度分批
    _cache = {}
    _roots = {}
    _mparts = {}
    if MONTHLY:                                                               # 分钟类根按月分批：每月一次分钟 SQL，算完该月所有根就释放
        _mins = [st for st in W if not st.startswith("L_")]
        for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
            _s = max(pd.Timestamp(start_date), _ms.start_time)
            _e = min(pd.Timestamp(end_date), _ms.end_time)
            if _s > _e:
                continue
            _c = {}
            for st in _mins:
                _mparts.setdefault(st, []).append(
                    _root_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"), st, _c))
            del _c
    for st in W:
        if st.startswith("L_"):
            warm = (pd.to_datetime(start_date) - pd.Timedelta(days=21)).strftime("%Y-%m-%d %H:%M:%S")
            x = _root_impl(datasources, warm, end_date, st, {})
            x["date"] = pd.to_datetime(x["date"])
            x = x.sort_values(["instrument", "date"]).reset_index(drop=True)
            x["factor"] = x.groupby("instrument", sort=False)["factor"].transform(
                lambda s: s.ewm(alpha=1 - ALPHA, adjust=False).mean())
            x = x[x["date"] >= pd.Timestamp(start_date)]
        elif MONTHLY:
            x = pd.concat(_mparts.pop(st), ignore_index=True)
            x["date"] = pd.to_datetime(x["date"])
        else:
            x = _root_impl(datasources, start_date, end_date, st, _cache)
            x["date"] = pd.to_datetime(x["date"])
        x["instrument"] = x["instrument"].astype(str)
        x["r_" + st] = x.groupby("date")["factor"].rank(pct=True) - 0.5
        _roots[st] = x[["date", "instrument", "r_" + st]]
    import gc
    gc.collect()                                                              # 先算完各根（只留很小的结果），最后才算头部：峰值≈头部本身
    if HEADQ:                                                                 # 头部按自然季度循环（与评测机按季度评分逐位一致）
        _hp = []
        for _q in pd.period_range(pd.Timestamp(start_date).to_period("Q"), pd.Timestamp(end_date).to_period("Q"), freq="Q"):
            _s = max(pd.Timestamp(start_date), _q.start_time)
            _e = min(pd.Timestamp(end_date), _q.end_time)
            if _s > _e:
                continue
            _hp.append(_head_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S")))
            gc.collect()
        _head = pd.concat(_hp, ignore_index=True).drop_duplicates(["date", "instrument"])
        del _hp
    else:
        _head = _head_impl(datasources, start_date, end_date)
    _head = _head.rename(columns={"factor": "fh"})
    _head["date"] = pd.to_datetime(_head["date"])
    _head["instrument"] = _head["instrument"].astype(str)
    _m = _head
    for st, x in _roots.items():
        _m = _m.merge(x, on=["date", "instrument"], how="left")
        _m["r_" + st] = _m["r_" + st].fillna(0.0)
    _m["factor"] = _m["fh"].astype("float64")
    for st, w in W.items():
        _m["factor"] = _m["factor"] + w * _m["r_" + st]

    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)].drop_duplicates(["date", "instrument"])
    _out = _pool.merge(_m[["date", "instrument", "factor"]].drop_duplicates(["date", "instrument"]), on=["date", "instrument"], how="left")
    _out["factor"] = _out["factor"].astype("float64").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _out.loc[_out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return _out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def _head_impl(datasources, start_date, end_date):
    """长窗口软同伴（240 日）+ 六特征 1.5 倍中性化（新增换手率与市值二次项）+（09:30 置零版）。

    148（换手率特征，0.834）与 149（市值二次项，0.834）各自与底座持平，这里同时放进中性化特征集，
      检验两个「官方风格未覆盖的暴露」一起剔除后能否超过 144。
    09:30 修复：同 104 号对照版本。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取数据：同伴相关系数需要月初之前 60 个交易日的日收益（约 90 个自然日，加上月内偏移与长假留足余量）。
    LOOKBACK_DAYS = 400
    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 240, 120, 2.0
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,
            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals
        FROM {bar1m}
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            pre_close,
            px_high,
            px_low,
            deals,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN amt ELSE 0
            END AS amt_c,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN vol ELSE 0
            END AS vol_c,
            CASE
                WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - px_low ELSE 0 END AS rng_c,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,
            CASE
                WHEN hms > '090000' AND hms <= '093000' THEN 93000
                WHEN hms > '093000' AND hms <= '100000' THEN 100000
                WHEN hms > '100000' AND hms <= '103000' THEN 103000
                WHEN hms > '103000' AND hms <= '110000' THEN 110000
                WHEN hms > '110000' AND hms <= '113000' THEN 113000
                WHEN hms > '130000' AND hms <= '133000' THEN 133000
                WHEN hms > '133000' AND hms <= '140000' THEN 140000
                WHEN hms > '140000' AND hms <= '143000' THEN 143000
                WHEN hms > '143000' AND hms <= '150000' THEN 150000
                ELSE -1
            END AS time_segment
        FROM cte_raw
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt_c) AS sum_amt,
            SUM(vol_c) AS sum_vol,
            SUM(body_c) AS sum_body,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low,
            SUM(rng_c) AS sum_rng,
            SUM(upsh_c) AS sum_upsh,
            SUM(deals) AS sum_deals
        FROM cte_snapshot
        WHERE time_segment != -1
        GROUP BY instrument_id, trading_day, time_segment
    )
    SELECT
        CAST(CONCAT(
            f.trading_day,
            ' ',
            strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        all_instruments.instrument,
        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)

    # 股票池：同伴只在股票池内挑选（向前取历史区间的成分，取不到时用评估区间的成分）。
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")
    pool_hist = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [query_start, end_date]},
        compression=True,
    ).df()
    pool_hist["date"] = pd.to_datetime(pool_hist["date"])
    pool_hist["instrument"] = pool_hist["instrument"].astype(str)
    stk_pool = pool_hist[(pool_hist["date"] >= pd.to_datetime(start_date)) & (pool_hist["date"] <= pd.to_datetime(end_date))]
    universe = set(pool_hist["instrument"].unique())
    df = df[df["instrument"].isin(universe)]
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()

    # 官方暴露表（日频）里的流通市值：用于构造「官方风格没有覆盖」的中性化特征。
    _exp = dai.query(
        "SELECT date, instrument, float_market_cap FROM %s" % exposure_table,
        filters={"date": [query_start, end_date]}, compression=True,
    ).df()
    _exp["day"] = pd.to_datetime(_exp["date"]).dt.normalize()
    _exp["instrument"] = _exp["instrument"].astype(str)
    _mcap = _exp.groupby(["instrument", "day"], sort=False)["float_market_cap"].last()
    # 暴露表第 t 天的流通市值按当天 15:00 收盘价计算，盘中截面不可得：一律取该股票前一交易日的值。
    _mcap = _mcap.sort_index().groupby(level=0).shift(1)
    _midx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["mcap"] = pd.Series(_mcap.reindex(_midx).to_numpy(), index=df.index)
    df["mcap"] = df["mcap"].where(df["mcap"] > 0)
    # 官方 10 风格暴露 + 行业码（与平台标签残差化口径一致），用于因子自身的中性化测试。
    _st = dai.query(
        "SELECT date, instrument, SIZE, BETA, MOMENTUM, RESVOL, SIZENL, BTOP, LIQUIDTY, EARNYILD, GROWTH, LEVERAGE, industry_level1_code FROM %s" % exposure_table,
        filters={"date": [query_start, end_date]}, compression=True,
    ).df()
    _st["day"] = pd.to_datetime(_st["date"]).dt.normalize()
    _st["instrument"] = _st["instrument"].astype(str)
    _st = _st.drop_duplicates(["instrument", "day"])
    # 风格暴露与行业同样取前一交易日的值（第 t 天的暴露在当天收盘后才可得）。
    _st = _st.sort_values(["instrument", "day"]).reset_index(drop=True)
    for _c in ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE", "industry_level1_code"]:
        _st[_c] = _st.groupby("instrument")[_c].shift(1)
    _st = _st.dropna(subset=["industry_level1_code"])
    _sg = _st.set_index(["instrument", "day"])
    _sid = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    for _c in ['SIZE', 'BETA', 'MOMENTUM', 'RESVOL', 'SIZENL', 'BTOP', 'LIQUIDTY', 'EARNYILD', 'GROWTH', 'LEVERAGE']:
        df["s_" + _c] = pd.Series(_sg[_c].reindex(_sid).to_numpy(), index=df.index)
    _ind = _sg["industry_level1_code"].astype(str)
    for _c in sorted(_ind.dropna().unique()):
        df["i_" + _c] = (_ind.reindex(_sid).to_numpy() == _c).astype(float)

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    def vwap_of(amt, vol):
        return (amt / vol.where(vol > 0)).where(amt > 0)

    def roll(s, k, fn, min_periods):
        return s.groupby(df["instrument"], sort=False).transform(lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())

    grp = df.groupby("instrument", sort=False)
    grp_day = df.groupby(["instrument", "day"], sort=False)
    same_day = grp["day"].shift(1) == df["day"]
    px = df["last_px"]

    day_vwap = vwap_of(grp_day["sum_amt"].cumsum(), grp_day["sum_vol"].cumsum())
    r_dvwap = safe_log_ratio(day_vwap, px)

    def rolling_vwap_rev(k):
        return safe_log_ratio(vwap_of(roll(df["sum_amt"], k, "sum", 1), roll(df["sum_vol"], k, "sum", 1)), px).fillna(r_dvwap)

    df["r_30"] = safe_log_ratio(grp["last_px"].shift(1).where(same_day, df["pre_close"]), px)
    df["r_rv4"] = rolling_vwap_rev(4)
    df["r_rv8"] = rolling_vwap_rev(8)
    df["r_rv16"] = rolling_vwap_rev(16)
    df["r_day"] = safe_log_ratio(df["pre_close"], px)

    # 个股 beta（40 窗口滚动回归，截断 [-1, 4]，不足取 1），用于特质波动。
    win_ret = safe_log_ratio(px, grp["last_px"].shift(1))
    ret30 = safe_log_ratio(px, grp["last_px"].shift(1).where(same_day)).fillna(safe_log_ratio(px, df["pre_close"]))
    mkt30 = ret30.groupby(df["date"]).transform("mean")
    ex, ey = roll(ret30, 40, "mean", 10), roll(mkt30, 40, "mean", 10)
    exy, ey2 = roll(ret30 * mkt30, 40, "mean", 10), roll(mkt30 ** 2, 40, "mean", 10)
    var_m = (ey2 - ey ** 2).where((ey2 - ey ** 2) > 1e-12)
    beta = ((exy - ex * ey) / var_m).clip(-1, 4).fillna(1.0)

    # 股票特征：规模（-ln 最近 40 窗口成交额）、特质波动（取负）、非流动性（Amihud）、低价、MAX（取负）。
    df["c_size"] = -np.log1p(roll(df["sum_amt"], 40, "sum", 10))
    idio = win_ret - beta * win_ret.groupby(df["date"]).transform("mean")
    df["c_ivol"] = -roll(idio, 40, "std", 10)
    df["c_illiq"] = roll(win_ret.abs() / (df["sum_amt"] / 1e8).where(df["sum_amt"] > 0), 40, "mean", 10)
    df["c_lowpx"] = -np.log(px.where(px > 0))

    # 换手率代理：最近 40 个窗口成交额 ÷ 流通市值（官方风格里没有直接的换手率暴露）。
    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]

    # 流通市值的非线性项：ln(流通市值) 的平方（官方 SIZE 风格只覆盖线性部分）。
    df["c_mcap2"] = np.log(df["mcap"]) ** 2
    df["c_max"] = -roll(win_ret, 40, "max", 10)
    # 分钟上影线占比：窗口内 Σ(最高-实体上沿)/Σ(最高-最低)，40 窗口均值取负（冲高回落越多，反转越强）。
    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)
    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)
    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())
    # 平均单笔成交额：Σ成交额/Σ委托笔数 的 40 窗口均值（大单/小单主导窗口的代理）。
    _avgtr = df["sum_amt"] / df["sum_deals"].where(df["sum_deals"] > 0)
    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)

    # 同伴：每个自然月月初，用此前 PEER_LOOKBACK 个交易日的日收益（收盘段价格 / 昨收）计算相关系数。
    daily = df.groupby(["day", "instrument"], sort=False).agg(px_last=("last_px", "last"), pc=("pre_close", "first"))
    daily_ret = safe_log_ratio(daily["px_last"], daily["pc"]).unstack("instrument").sort_index()
    peer_insts = daily_ret.columns.values
    daily_amt = df.groupby(["day", "instrument"], sort=False)["sum_amt"].sum().unstack("instrument").reindex(columns=peer_insts).sort_index()
    all_days = daily_ret.index
    day_month = all_days.to_period("M")
    peers_by_month = {}
    out_months = pd.to_datetime(stk_pool["date"]).dt.normalize().dt.to_period("M").unique()
    for m in out_months:
        in_month = all_days[day_month == m]
        if len(in_month) == 0:
            continue
        hist = daily_ret.loc[daily_ret.index < in_month[0]].tail(PEER_LOOKBACK)
        if len(hist) < PEER_MIN_OBS:
            continue
        ok = (hist.notna().sum() >= PEER_MIN_OBS).values
        corr = hist.loc[:, ok].corr(min_periods=PEER_MIN_OBS).values
        np.fill_diagonal(corr, np.nan)
        corr = np.nan_to_num(corr, nan=-1.0)          # 缺失相关性记 -1（等价于不选）
        sub = np.where(ok)[0]
        top = np.argsort(-corr, axis=1, kind="stable")[:, :PEER_K]
        idx = np.full((len(peer_insts), PEER_K), -1, dtype=np.int64)
        idx[sub] = sub[top]
        cvals = corr[np.arange(len(corr))[:, None], top]
        # 规模加权：相关度^p × 该同伴前 PEER_LOOKBACK 日的平均成交额（大票的共同运动信息权重更高）。
        _aw = daily_amt.loc[daily_amt.index < in_month[0]].tail(PEER_LOOKBACK).mean().fillna(0.0).values
        _aw = np.where(_aw > 0, _aw, 0.0)
        wts = (np.clip(cvals, 0.0, None) ** PEER_POWER) * _aw[sub][:, None]
        wsum = wts.sum(axis=1, keepdims=True)
        wts = np.where(wsum > 0, wts / np.where(wsum > 0, wsum, 1.0), 0.0)
        full_w = np.zeros((len(peer_insts), PEER_K))
        full_w[sub] = wts
        peers_by_month[m] = (idx, full_w)

    df = df.replace([np.inf, -np.inf], np.nan)
    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]
    keep = keep + sorted([c for c in df.columns if c.startswith("i_")])
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool[["date", "instrument"]], how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)
    result_month = result["date"].dt.to_period("M")

    def peer_mean(col):
        """每个截面上同伴的 col 均值（至少 3 个有效值），没有同伴信息时为缺失。"""
        out = np.full(len(result), np.nan)
        for m, rows in result.groupby(result_month, sort=False).indices.items():
            rec = peers_by_month.get(m)
            if rec is None:
                continue
            idx, wts = rec
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            valid = idx >= 0
            g = wide.values[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            ww = np.where(valid[None, :, :], wts[None, :, :], 0.0)
            okm = ~np.isnan(g)
            cnt = okm.sum(axis=2)
            wsum = (ww * okm).sum(axis=2)
            pm = np.where((cnt >= 3) & (wsum > 0), np.nansum(np.where(okm, g, 0.0) * ww, axis=2) / np.where(wsum > 0, wsum, 1.0), np.nan)
            r_pos = wide.index.get_indexer(sub["date"])
            c_pos = pd.Index(peer_insts).get_indexer(sub["instrument"])
            out[rows] = np.where(c_pos >= 0, pm[r_pos, np.maximum(c_pos, 0)], np.nan)
        return pd.Series(out, index=result.index)

    def rank_blend(cols, weights=None):
        """各分量在股票池内逐截面居中排名后加权平均（缺失分量不参与，权重按可用分量重新归一）。"""
        weights = weights or [1.0] * len(cols)
        ranks = (result.groupby("date")[cols].rank(pct=True) - 0.5).to_numpy()
        w = np.array(weights, dtype=float)
        has = ~np.isnan(ranks)
        den = (has * w).sum(axis=1)
        num = np.nansum(np.nan_to_num(ranks) * w, axis=1)
        return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=result.index)

    def neutralize(col, chars, frac=1.0):
        """截面中性化：col 的居中排名对各特征居中排名（缺失记 0）逐截面 OLS（含截距），返回 排名 - frac x 拟合值。"""
        y = result.groupby("date")[col].rank(pct=True) - 0.5
        X = np.column_stack([(result.groupby("date")[c].rank(pct=True) - 0.5).fillna(0.0).values for c in chars])
        ok = y.notna().values
        codes, uniq = pd.factorize(result["date"].values)
        k, n_grp = X.shape[1], len(uniq)
        Xc = np.column_stack([np.ones(len(result)), X]) * ok[:, None]
        yv = np.nan_to_num(y.values) * ok
        xtx = np.zeros((n_grp, k + 1, k + 1))
        xty = np.zeros((n_grp, k + 1))
        for i in range(k + 1):
            xty[:, i] = np.bincount(codes, Xc[:, i] * yv, minlength=n_grp)
            for j in range(i, k + 1):
                v = np.bincount(codes, Xc[:, i] * Xc[:, j], minlength=n_grp)
                xtx[:, i, j] = v
                xtx[:, j, i] = v
        b = np.linalg.solve(xtx + 1e-9 * np.eye(k + 1), xty[..., None])[..., 0]
        fit = (X * b[codes, 1:]).sum(axis=1)
        return pd.Series(np.where(ok, y.values - frac * fit, np.nan), index=result.index)

    def peer_ladder():
        cols = []
        for c in ["r_30", "r_rv4", "r_rv8", "r_day"]:
            result[c + "_p"] = result[c] - peer_mean(c).fillna(result.groupby("date")[c].transform("mean"))
            cols.append(c + "_p")
        return rank_blend(cols)

    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"] + sorted([c for c in df.columns if c.startswith("i_")])
    result["ladder"] = peer_ladder()
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])
    result["factor"] = neutralize("signal", CHARS4, 2.5)
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面。
    result.loc[result["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def _root_impl(datasources, start_date, end_date, STAT, _cache):
    """【新根因子 第六轮 L：资金流窗口框架（1340 起）】一份代码覆盖 L/M/N/O/P/Q/R/S/T/U/V 十一个方向，STAT 选一个统计量。
    同一段 SQL 被所有变体共用（评测机按 SQL 文本缓存），变体只改 STAT 一行。

    分钟口径：1 分钟 bar 用结束时刻标注，截面 T 取标签恰为 T 的那条 bar（含）之前的连续竞价分钟，
    不含集合竞价 bar（09:30 及之前）；"当日至今"的量都按 (股票, 交易日) 分区、按时间累计，不跨日。
    30 分钟段口径：(09:30,10:00] … (14:30,15:00]，只用截面所在交易日之前的完整交易日（多日统计）
    或截面时刻及之前的段（T_acc）。不读 vwap 表、不读暴露表、不含自身收益反转写法。

    STAT：
      L_lab10/20/40  下一个半小时段在过去 N 日的"段收盘 ÷ 段 VWAP − 1"均值（同时段季节性）
      L_ret20        下一个半小时段在过去 20 日的段收益均值；L_lab20_dm 去掉各时段平均
      M_q / M_q_lin  聪明钱：以 |r|/√v（或 |r|/v）加权的当日 VWAP ÷ 当日 VWAP − 1
      N_dir / N_big  大单方向：Σ r·(ln 单笔额 − 之前分钟均值) ÷ Σ|r|；大单分钟收益和 − 小单分钟收益和
      Q_eff / Q_seff / Q_ac1  路径效率 |净变动|÷Σ|r|、带符号版、分钟收益一阶自相关
      U_nj / U_sj    当日跳跃次数（|r| > 4 倍之前分钟波动）与带符号跳跃收益和
      S_osz / S_osz30  买方每笔委托挂单量 ÷ 卖方（1–3 档，对数）当日均值 / 近 30 分钟均值
      O_q5           一档买量 ÷ 卖量（对数）近 5 分钟均值
      T_acc          本段成交额 ÷ 上一段（对数）× 本段收益符号
      R_round / R_half  当前价相对最近整数元 / 半元的带符号距离（÷ 价格）
      V_spr          当日相对价差均值
      W_vdir         量价同向：Σ r·(ln 成交量 − 之前分钟均值) ÷ Σ|r|
      W_cnt          1–3 档委托笔数不平衡 (买 − 卖)/(买 + 卖) 的当日均值
      W_hhi          当日分钟成交量集中度 Σv² ÷ (Σv)²
      W_zero         当日零收益分钟占比（价格黏滞 / 跳价约束）
      W_atsd         当日分钟平均单笔成交额（对数）的离散度
      W_hl           Σ(分钟最高 − 最低)/收盘 ÷ |当日净变动|（日内往返程度）
      X_q60 / X_q30  近 60 / 30 分钟带符号路径效率；X_q5d 近 5 日（1200 分钟）带符号路径效率
      X_n5d / X_m5d  近 5 日大单方向 / 聪明钱（|r|/v 加权 VWAP ÷ VWAP − 1）
      X_m75 / X_ma   聪明钱权重 |r|/v^0.75、|r|/成交额
      L_rest20       当日剩余各时段过去 20 日"收盘 ÷ VWAP − 1"均值的平均（季节性，换手更低）
      P_ac20 / P_ac60  个股过去 20/60 日相邻半小时段收益的一阶自相关（日内常数，供条件化头部用）
    """
    import numpy as np
    import pandas as pd
    import dai

    SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
    NEXT = {"1000": "1030", "1030": "1100", "1100": "1130", "1130": "1330", "1330": "1400", "1400": "1430", "1430": "1500"}
    bar1m = datasources["bar1m"]
    fam = STAT.split("_")[0]

    if fam in ("M", "N", "Q", "U", "S", "O", "R", "V", "W", "X", "Y", "Z"):
        sql_min = f"""
        WITH b AS (
            SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d, strftime(b.date, '%H%M') AS hm,
                CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c, CAST(b.high AS DOUBLE) AS h, CAST(b.low AS DOUBLE) AS l,
                CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
                CAST(COALESCE(b.deal_number, 0) AS DOUBLE) AS n,
                CAST(COALESCE(b.bid_volume1, 0) + COALESCE(b.bid_volume2, 0) + COALESCE(b.bid_volume3, 0) AS DOUBLE) AS bv,
                CAST(COALESCE(b.ask_volume1, 0) + COALESCE(b.ask_volume2, 0) + COALESCE(b.ask_volume3, 0) AS DOUBLE) AS av,
                CAST(COALESCE(b.bid_num_orders1, 0) + COALESCE(b.bid_num_orders2, 0) + COALESCE(b.bid_num_orders3, 0) AS DOUBLE) AS bo,
                CAST(COALESCE(b.ask_num_orders1, 0) + COALESCE(b.ask_num_orders2, 0) + COALESCE(b.ask_num_orders3, 0) AS DOUBLE) AS ao,
                CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1, CAST(COALESCE(b.ask_volume1, 0) AS DOUBLE) AS av1,
                CAST(b.bid_price1 AS DOUBLE) AS bp1, CAST(b.ask_price1 AS DOUBLE) AS ap1
            FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
            WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '150000'
              AND b.close > 0 AND b.open > 0
        ), m1 AS (
            SELECT *, LN(c / COALESCE(LAG(c) OVER (PARTITION BY instrument_id, d ORDER BY date), o)) AS r,
                CASE WHEN n > 0 AND a > 0 THEN LN(a / n) END AS lats, CASE WHEN v > 0 THEN LN(v) END AS lv
            FROM b
        ), m2 AS (
            SELECT *, ABS(r) / GREATEST(v, 1) AS s2,
                AVG(lats) OVER (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS lats_prev,
                AVG(lv) OVER (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS lv_prev
            FROM m1
        ), m3 AS (
            SELECT instrument_id, instrument, d, hm, date, c,
                SUM(a) OVER w AS ca, SUM(v) OVER w AS cv, SUM(s2 * a) OVER w AS cs2a, SUM(s2 * v) OVER w AS cs2v,
                SUM(r * (lats - lats_prev)) OVER w AS cnd,
                SUM(ABS(r)) OVER w AS cabs,
                SUM(r * (lv - lv_prev)) OVER w AS cvd, SUM(CASE WHEN r = 0 THEN 1 ELSE 0 END) OVER w AS czero, COUNT(r) OVER w AS cnr, SUM(ABS(r)) OVER w5 AS a5d, SUM(r * (lats - lats_prev)) OVER w5 AS nd5, STDDEV_SAMP(lats) OVER w5 AS sd5,
                SUM(CASE WHEN r > 0 THEN v ELSE 0 END) OVER w60 AS vup60, SUM(v) OVER w60 AS v60,
                SUM(CASE WHEN r > 0 THEN v ELSE 0 END) OVER w90 AS vup90, SUM(v) OVER w90 AS v90
            FROM m2
            WINDOW w AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                   w60 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
                   w30 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),
                   w5 AS (PARTITION BY instrument_id ORDER BY date ROWS BETWEEN 1199 PRECEDING AND CURRENT ROW),
                   w90 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW),
                   w120 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW),
                   w60x AS (PARTITION BY instrument_id ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        )
        SELECT * FROM m3 WHERE hm IN ('1000', '1030', '1100', '1130', '1330', '1400', '1430')
        """
        q5 = (pd.to_datetime(start_date) - pd.Timedelta(days=12)).strftime("%Y-%m-%d %H:%M:%S")     # 近 5 日窗口需要之前的分钟
        if "min" not in _cache:                                                    # 同一调用内只查一次
            _cache["min"] = dai.query(sql_min, filters={"date": [q5, end_date]}, compression=True).df()
        df = _cache["min"].copy()
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(start_date)].reset_index(drop=True)
        if not len(df):
            return pd.DataFrame(columns=["date", "instrument", "factor"])
        with np.errstate(all="ignore"):
            vw = df["ca"] / df["cv"].where(df["cv"] > 0)
            cabs = df["cabs"].where(df["cabs"] > 0)
            sig = {
                "M_q": lambda: (df["cs1a"] / df["cs1v"].where(df["cs1v"] > 0)) / vw - 1,
                "M_q_lin": lambda: (df["cs2a"] / df["cs2v"].where(df["cs2v"] > 0)) / vw - 1,
                "N_dir": lambda: df["cnd"] / cabs,
                "N_big": lambda: df["cbig"] - df["csmall"],
                "Q_eff": lambda: df["csum"].abs() / cabs,
                "Q_seff": lambda: df["csum"] / cabs,
                "Q_ac1": lambda: df["crr1"] / df["crr"].where(df["crr"] > 0),
                "U_nj": lambda: df["cnj"].astype(float),
                "U_sj": lambda: df["csj"],
                "S_osz": lambda: df["mosz"],
                "S_osz30": lambda: df["mosz30"],
                "O_q5": lambda: df["mq5"],
                "R_round": lambda: (df["c"] - np.round(df["c"])) / df["c"],
                "R_half": lambda: (df["c"] - np.round(df["c"] * 2) / 2) / df["c"],
                "V_spr": lambda: df["mspr"],
                "W_vdir": lambda: df["cvd"] / cabs,
                "X_q60": lambda: df["r60"] / df["a60"].where(df["a60"] > 0),
                "X_q30": lambda: df["r30"] / df["a30"].where(df["a30"] > 0),
                "X_q5d": lambda: df["r5d"] / df["a5d"].where(df["a5d"] > 0),
                "X_n5d": lambda: df["nd5"] / df["a5d"].where(df["a5d"] > 0),
                "X_m5d": lambda: (df["s2a_5"] / df["s2v_5"].where(df["s2v_5"] > 0)) / (df["a_5"] / df["v_5"].where(df["v_5"] > 0)) - 1,
                "X_m75": lambda: (df["cs3a"] / df["cs3v"].where(df["cs3v"] > 0)) / vw - 1,
                "X_ma": lambda: (df["cs4a"] / df["cs4v"].where(df["cs4v"] > 0)) / vw - 1,
                "Y_vd5": lambda: df["vd5"] / df["a5d"].where(df["a5d"] > 0),
                "Y_z5": lambda: df["z5"] / df["n5"].where(df["n5"] > 0),
                "Y_sd5": lambda: df["sd5"],
                "Y_mfr": lambda: df["vup"] / df["vall"].where(df["vall"] > 0),
                "Y_bid": lambda: df["abid"] / df["aall"].where(df["aall"] > 0),
                "Z_aup": lambda: df["aup"] / df["aall"].where(df["aall"] > 0),
                "Z_mfr60": lambda: df["vup60"] / df["v60"].where(df["v60"] > 0),
                "Z_ats": lambda: np.log((df["aup"] / df["nup"].where(df["nup"] > 0)) / (df["adn"] / df["ndn"].where(df["ndn"] > 0))),
                "Z_mfr30": lambda: df["vup30"] / df["v30"].where(df["v30"] > 0),
                "Z_mfr90": lambda: df["vup90"] / df["v90"].where(df["v90"] > 0),
                "Z_mfr120": lambda: df["vup120"] / df["v120"].where(df["v120"] > 0),
                "Z_net60": lambda: (df["vup60"] - df["vdn60"]) / df["v60"].where(df["v60"] > 0),
                "Z_amfr60": lambda: df["aup60"] / df["am60"].where(df["am60"] > 0),
                "Z_mfr60x": lambda: df["vup60x"] / df["v60x"].where(df["v60x"] > 0),
                "W_cnt": lambda: df["mcimb"],
                "W_hhi": lambda: df["cvv"] / (df["cv"] ** 2).where(df["cv"] > 0),
                "W_zero": lambda: df["czero"] / df["cnr"].where(df["cnr"] > 0),
                "W_atsd": lambda: df["sdlats"],
                "W_hl": lambda: df["chl"] / df["csum"].abs().where(df["csum"].abs() > 0),
            }[STAT.split("_res")[0]]()
        df["signal"] = np.asarray(sig, dtype=float)
        if "_res" in STAT:                                                   # 逐截面对当日收益（_res2 再加量价同向）回归取残差
            Xr = [df["csum"].to_numpy(float)]
            if STAT.endswith("_res2"):
                Xr.append((df["cvd"] / cabs).to_numpy(float))
            Xr = np.column_stack(Xr)
            y_ = df["signal"].to_numpy(float)
            out_ = np.full(len(df), np.nan)
            for _, ii in df.groupby("date").indices.items():
                ok_ = np.isfinite(y_[ii]) & np.isfinite(Xr[ii]).all(1)
                if ok_.sum() < 50:
                    continue
                A_ = np.column_stack([np.ones(ok_.sum()), Xr[ii][ok_]])
                coef_, *_ = np.linalg.lstsq(A_, y_[ii][ok_], rcond=None)
                out_[ii[ok_]] = y_[ii][ok_] - A_ @ coef_
            df["signal"] = out_
        df = df[["date", "instrument", "signal"]]
    else:
        q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=130)).strftime("%Y-%m-%d %H:%M:%S")
        sql_seg = f"""
        WITH b AS (
            SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d,
                CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c,
                CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
                CASE
                    WHEN strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '100000' THEN '1000'
                    WHEN strftime(b.date, '%H%M%S') > '100000' AND strftime(b.date, '%H%M%S') <= '103000' THEN '1030'
                    WHEN strftime(b.date, '%H%M%S') > '103000' AND strftime(b.date, '%H%M%S') <= '110000' THEN '1100'
                    WHEN strftime(b.date, '%H%M%S') > '110000' AND strftime(b.date, '%H%M%S') <= '113000' THEN '1130'
                    WHEN strftime(b.date, '%H%M%S') > '130000' AND strftime(b.date, '%H%M%S') <= '133000' THEN '1330'
                    WHEN strftime(b.date, '%H%M%S') > '133000' AND strftime(b.date, '%H%M%S') <= '140000' THEN '1400'
                    WHEN strftime(b.date, '%H%M%S') > '140000' AND strftime(b.date, '%H%M%S') <= '143000' THEN '1430'
                    WHEN strftime(b.date, '%H%M%S') > '143000' AND strftime(b.date, '%H%M%S') <= '150000' THEN '1500'
                END AS seg
            FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
            WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '150000'
              AND b.close > 0 AND b.open > 0
        )
        SELECT MAX(date) AS date, instrument_id, MAX(instrument) AS instrument, d, seg,
            SUM(a) AS amt, SUM(v) AS vol, arg_max(c, date) AS c_last, arg_min(o, date) AS o_first
        FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
        """
        sg = dai.query(sql_seg, filters={"date": [q0, end_date]}, compression=True).df()
        if not len(sg):
            return pd.DataFrame(columns=["date", "instrument", "factor"])
        sg["instrument"] = sg["instrument"].astype(str)
        sg["d"] = pd.to_datetime(sg["d"])
        sg = sg.sort_values(["instrument", "d", "seg"]).reset_index(drop=True)
        with np.errstate(all="ignore"):
            sg["lab"] = sg["c_last"] / (sg["amt"] / sg["vol"].where(sg["vol"] > 0)) - 1
            prev_c = sg.groupby(["instrument", "d"])["c_last"].shift(1)
            sg["ret"] = sg["c_last"] / prev_c.fillna(sg["o_first"]) - 1
        days = np.sort(sg["d"].unique())
        out_days = days[days >= pd.Timestamp(pd.to_datetime(start_date).normalize())]
        rows = []
        if fam == "L":
            col = "ret" if STAT.startswith("L_ret") else "lab"
            N = int("".join(ch for ch in STAT.split("_")[1] if ch.isdigit()))
            piv = sg.pivot_table(index="d", columns=["instrument", "seg"], values=col, aggfunc="last").reindex(days)
            mu = piv.rolling(N, min_periods=max(N // 2, 3)).mean().shift(1)       # 只用之前的交易日
            if STAT.startswith("L_rest"):                                          # 当日剩余时段的平均季节性
                segs = ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
                for t in SLOTS:
                    rest = segs[segs.index(NEXT[t]):]
                    s = mu.loc[out_days].loc[:, mu.columns.get_level_values(1).isin(rest)].T.groupby(level=0).mean().T
                    s = s.stack().rename("signal").reset_index()
                    s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
                    rows.append(s[["date", "instrument", "signal"]])
            if STAT.endswith("_dm"):
                mu = mu - mu.T.groupby(level=0).transform("mean").T
            for t in ([] if STAT.startswith("L_rest") else SLOTS):
                nx = NEXT[t]
                s = mu.loc[out_days].xs(nx, axis=1, level=1)
                s = s.stack().rename("signal").reset_index()
                s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
                rows.append(s[["date", "instrument", "signal"]])
        elif fam == "P":
            N = int(STAT.split("ac")[1])
            sg["ret_next"] = sg.groupby(["instrument", "d"])["ret"].shift(-1)
            g = sg.assign(xy=sg["ret"] * sg["ret_next"], xx=sg["ret"] ** 2).groupby(["d", "instrument"])[["xy", "xx"]].sum()
            xy = g["xy"].unstack().reindex(days)
            xx = g["xx"].unstack().reindex(days)
            ac = (xy.rolling(N, min_periods=N // 2).sum() / xx.rolling(N, min_periods=N // 2).sum()).shift(1)
            s0 = ac.loc[out_days].stack().rename("signal").reset_index()
            for t in SLOTS:
                s = s0.copy()
                s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
                rows.append(s[["date", "instrument", "signal"]])
        else:                                                                      # T_acc
            prev_amt = sg.groupby("instrument")["amt"].shift(1)
            with np.errstate(all="ignore"):
                sg["signal"] = np.log(sg["amt"] / prev_amt.where(prev_amt > 0)) * np.sign(sg["ret"])
            s = sg[sg["seg"].isin(SLOTS) & sg["d"].isin(out_days)].copy()
            s["date"] = s["d"] + pd.to_timedelta(s["seg"].str[:2].astype(int), "h") + pd.to_timedelta(s["seg"].str[2:].astype(int), "m")
            rows.append(s[["date", "instrument", "signal"]])
        df = pd.concat(rows, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    df["signal"] = df["signal"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["instrument", "signal"]).drop_duplicates(["date", "instrument"])
    df["factor"] = df.groupby("date")["signal"].rank(pct=True) - 0.5
    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)].drop_duplicates(["date", "instrument"])
    out = _pool.merge(df[["date", "instrument", "factor"]], on=["date", "instrument"], how="left")
    out["factor"] = out["factor"].astype("float64").fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


if __name__ == "__main__":
    import pandas as pd
    from bigmodule import M
    import structlog

    logger = structlog.get_logger()
    datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
    start_date = "2020-01-01 00:00:00"
    end_date = "2020-03-01 23:59:59"
    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
