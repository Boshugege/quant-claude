def main(datasources, start_date, end_date):
    """行业内前一日 BETA 相近同伴残差动量（方向已按 581 实测 IC=−0.0720 校正取正号；暴露表错一天）。

    长窗口软同伴（240 日）+ 分钟上影线占比附加（0.15）+ 四特征 1.5 倍中性化（09:30 置零版，上影线权重 0.15 → 0.30。）。

    叠加两个各自刚验证有效的杠杆：144（同伴相关性窗口 240 日，0.842）与 153（分钟上影线占比，0.834）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取数据：同伴相关系数需要月初之前 60 个交易日的日收益（约 90 个自然日，加上月内偏移与长假留足余量）。
    LOOKBACK_DAYS = 400
    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 240, 120, 3.0
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
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
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
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN LEAST(px_open, px_close) - px_low ELSE 0 END AS dnsh_c,
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
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low,
            SUM(rng_c) AS sum_rng,
            SUM(upsh_c) AS sum_upsh,
            SUM(dnsh_c) AS sum_dnsh,
            SUM(body_c) AS sum_body,
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
        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,
        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",
              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)

    # 股票池：同伴只在股票池内挑选（向前取历史区间的成分，取不到时用评估区间的成分）。
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
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

    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")
    # 暴露表错位：第 t 天的市值/风格/行业按当天 15:00 收盘计算，盘中截面不可得；
    # 一律取该股票「严格早于截面日期的最近一条」（前一交易日及更早）。
    _exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
    _exp = dai.query("SELECT date, instrument, " + ", ".join(_exp_cols) + " FROM " + exposure_table,
                     filters={"date": [query_start, end_date]}, compression=True).df()
    _exp["day"] = pd.to_datetime(_exp["date"]).dt.normalize()
    _exp["instrument"] = _exp["instrument"].astype(str)
    _exp = _exp.sort_values(["instrument", "day"]).drop_duplicates(["instrument", "day"]).reset_index(drop=True)
    for _c in _exp_cols:
        _exp[_c] = _exp.groupby("instrument", sort=False)[_c].shift(1)
    _exp = _exp.set_index(["instrument", "day"])
    _midx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["mcap"] = pd.Series(_exp["float_market_cap"].reindex(_midx).to_numpy(), index=df.index)
    df["mcap"] = df["mcap"].where(df["mcap"] > 0)
    for _c in ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]:
        df["s_" + _c] = pd.Series(_exp[_c].reindex(_midx).to_numpy(), index=df.index)
    _ind = _exp["industry_level1_code"].reindex(_midx).astype(str)
    df["ind_code"] = _ind.to_numpy()

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

    # 分钟上影线占比：窗口内 Σ(最高-实体上沿) / Σ(最高-最低)，最近 40 窗口均值取负（冲高回落越多，反转越强）。
    # ===== 新增轴：按前一日风格暴露分组的条件反转（LIQUIDTY 三分位内去均值） =====
    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正

    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _beta66 = df["s_BETA"]
    _qb66 = ((_beta66.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _qb66.values]
    _res66 = []
    for _c66 in ["r_30", "r_rv4", "r_rv8", "r_day"]:
        _v66 = df[_c66]
        _ok66 = (_v66.notna() & (_mkw > 0)).to_numpy()
        _vk = np.where(_ok66, np.nan_to_num(_v66.to_numpy()), 0.0)
        _wk = np.where(_ok66, _mkw.to_numpy(), 0.0)
        _gsum = pd.Series(_vk * _wk, index=df.index).groupby(_pk66).transform("sum").to_numpy()
        _gw = pd.Series(_wk, index=df.index).groupby(_pk66).transform("sum").to_numpy()
        _self66 = _vk * _wk
        _den66 = _gw - _wk
        _peer66 = np.where(_den66 > 0, (_gsum - _self66) / np.where(_den66 > 0, _den66, 1.0), np.nan)
        _res66.append(pd.Series(_v66.to_numpy() - _peer66, index=df.index))
    _rkb = pd.concat([x.groupby(df["date"]).rank(pct=True) for x in _res66], axis=1).mean(axis=1) - 0.5
    df["c_indpeer"] = _rkb

    # ===== 新增轴：真换手率相对自身 20 日历史 =====
    _dly66 = df.groupby(["day", "instrument"], sort=False).agg(
        _mcap=("mcap", "first"), _amt=("sum_amt", "sum"), _ind=("ind_code", "first")).reset_index()
    _dly66["_turn"] = _dly66["_amt"] / _dly66["_mcap"].where(_dly66["_mcap"] > 0)
    _dly66 = _dly66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dly66["_tm20"] = _dly66.groupby("instrument", sort=False)["_turn"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    _dly66["_turnself"] = np.log(_dly66["_turn"] / _dly66["_tm20"].where(_dly66["_tm20"] > 0))
    _m66 = _dly66.set_index(["instrument", "day"])
    _key66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["c_turnself"] = -_m66["_turnself"].reindex(_key66).to_numpy()
    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)
    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())
    # ===== 新增轴：行业滞后（前一日）收益反转 =====
    _dly66 = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first"), _ind=("ind_code", "first")).reset_index()
    _dly66["_r"] = np.where((_dly66["_last"] > 0) & (_dly66["_pc"] > 0), np.log(_dly66["_last"] / _dly66["_pc"]), np.nan)
    _idp = _dly66.groupby(["day", "_ind"], sort=False)["_r"].mean().unstack("_ind").sort_index().shift(1)
    _ikey = pd.MultiIndex.from_arrays([df["day"].values, df["ind_code"].values])
    _flat = _idp.stack(dropna=False)
    _flat.index = pd.MultiIndex.from_arrays([_flat.index.get_level_values(0), _flat.index.get_level_values(1)])
    df["c_indrev"] = -_flat.reindex(_ikey).to_numpy()
    # 实体占比（分钟 |收-开| 之和 / 振幅之和），40 窗口均值取负。
    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)
    # 平均单笔成交额：Σ成交额/Σ委托笔数，40 窗口均值（大小单主导的代理）。
    _avgtr = df["sum_amt"] / df["sum_deals"].where(df["sum_deals"] > 0)
    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)
    # 委托笔数（log）：窗口内成交笔数的 40 窗口均值，活跃度/散户参与度代理。
    df["c_deals"] = roll(np.log1p(df["sum_deals"]), 40, "mean", 10)
    df["c_max"] = -roll(win_ret, 40, "max", 10)

    df = df.replace([np.inf, -np.inf], np.nan)
    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]
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

    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    _ord98 = result.sort_values(["instrument", "date"])
    result["signal"] = _ord98.groupby("instrument", sort=False)["c_indpeer"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)
    result["factor"] = result["signal"]
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面。
    result.loc[result["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


if __name__ == "__main__":
    from bigmodule import M
    import structlog

    logger = structlog.get_logger()
    datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m"}
    start_date = "2020-01-01 00:00:00"
    end_date = "2020-03-01 23:59:59"
    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
