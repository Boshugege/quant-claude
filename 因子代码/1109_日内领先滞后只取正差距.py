def main(datasources, start_date, end_date):
    """【新根因子 方向B：日内领先滞后】前日同行业内市值前 20% 的"领先股"上一段收益均值 − 本股上一段收益。
    【方向B变体 1109：日内领先滞后只取正差距】

    假设：大盘股／行业龙头先反应新信息，同一行业内的小票随后"追赶"；因此"领先组刚走过的这一段收益"
    减去"自己刚走过的这一段收益"是尚未被自己反映的追赶压力，对之后 30 分钟的收益有预测力。

    只用到：1 分钟行情的收价与前收（自身收益只作为**减数**）、暴露表**前一交易日**的行业与流通市值（显式 shift(1)）。
    不含反转阶梯、不含 K 线形态分量、不含 c_utime/c_hitime/c_lotime。分段聚合在 SQL 里做（内存友好）。

    参数：LEAD_GROUP=ind_top|mkt_top|ind_wavg（领先组定义）、LEAD_TOPN=领先组市值分位、
    LEAD_LAG=用第几段之前的收益（1=上一段）、CATCH_ONLY=是否只取正差距、MIN_PEERS=领先组最少只数。
    """
    import numpy as np
    import pandas as pd
    import dai

    LEAD_GROUP = "ind_top"   # ind_top | mkt_top | ind_wavg
    LEAD_TOPN = 0.20         # 领先组取前多少比例（按前一日流通市值）
    LEAD_LAG = 1             # 1 = 上一段（t-1→t），2 = 上两段
    CATCH_ONLY = 1           # 1 = 只取 max(0, 领先组收益 − 自己收益)
    MIN_PEERS = 3            # 领先组最少只数，不足记缺失

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S")
    instr_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close
        FROM {bar1m} b
        LEFT JOIN all_instruments ai USING (instrument_id)
    ),
    cte_seg AS (
        SELECT
            instrument_id, instrument, trading_day, hms, px_close, pre_close,
            CASE WHEN px_close > 0 THEN px_close ELSE 0 END AS px,
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
    )
    SELECT
        CAST(CONCAT(
            s.trading_day, ' ',
            strftime(strptime(LPAD(CAST(s.time_segment AS VARCHAR), 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        s.trading_day,
        s.time_segment,
        s.instrument_id,
        MAX(s.instrument) AS instrument,
        MAX(CASE WHEN CAST(s.hms AS INTEGER) = s.time_segment
                  OR (s.time_segment = 93000 AND s.hms = '092500') THEN s.px ELSE 0 END) AS last_px,
        MAX(s.pre_close) AS pre_close
    FROM cte_seg s
    WHERE s.time_segment != -1
    GROUP BY s.trading_day, s.time_segment, s.instrument_id
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    if not len(df):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    df["instrument_id"] = df["instrument_id"].astype(str)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df = df.sort_values(["instrument_id", "trading_day", "time_segment"]).reset_index(drop=True)

    df["instrument"] = df["instrument"].astype(str)

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    gs = df.groupby(["instrument_id", "trading_day"], sort=False)
    px_series = df["last_px"].where(df["last_px"] > 0)
    day_pre = gs["pre_close"].transform("max")
    px_lag1 = px_series.groupby([df["instrument_id"], df["trading_day"]], sort=False).shift(1)
    px_lag1 = px_lag1.where(px_lag1 > 0, day_pre)
    px_lag2 = px_series.groupby([df["instrument_id"], df["trading_day"]], sort=False).shift(2)
    df["ret1"] = safe_log_ratio(px_series, px_lag1)
    df["ret2"] = safe_log_ratio(px_lag1, px_lag2)
    df["seg_ret"] = df["ret1"] if LEAD_LAG == 1 else df["ret2"]

    # 暴露表错位：一律取严格早于截面日期的最近一条
    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")
    _exp_cols = ["float_market_cap", "industry_level1_code"]
    _exp = dai.query("SELECT date, instrument, " + ", ".join(_exp_cols) + " FROM " + exposure_table,
                     filters={"date": [query_start, end_date]}, compression=True).df()
    _exp["day"] = pd.to_datetime(_exp["date"]).dt.normalize()
    _exp["instrument"] = _exp["instrument"].astype(str)
    _exp = _exp.sort_values(["instrument", "day"]).drop_duplicates(["instrument", "day"]).reset_index(drop=True)
    for _c in _exp_cols:
        _exp[_c] = _exp.groupby("instrument", sort=False)[_c].shift(1)
    _exp = _exp.set_index(["instrument", "day"])
    _midx = pd.MultiIndex.from_arrays([df["instrument"].values, df["trading_day"].values])
    df["mcap"] = pd.Series(_exp["float_market_cap"].reindex(_midx).to_numpy(), index=df.index)
    df["mcap"] = df["mcap"].where(df["mcap"] > 0)
    df["ind"] = pd.Series(_exp["industry_level1_code"].reindex(_midx).to_numpy(), index=df.index).astype(str)

    keys = ["trading_day", "time_segment"] if LEAD_GROUP == "mkt_top" else ["trading_day", "time_segment", "ind"]
    work = df[df["seg_ret"].notna() & df["mcap"].notna() & (df["ind"] != "nan")].copy()
    if LEAD_GROUP == "ind_wavg":
        work["_w"] = work["mcap"].to_numpy()
    else:
        q = work.groupby(keys, sort=False)["mcap"].transform(lambda x: x.quantile(1.0 - LEAD_TOPN))
        work["_w"] = (work["mcap"] >= q).astype(float)
        work["_w"] = work["_w"].where(work.groupby(keys, sort=False)["_w"].transform("sum") > 1, 0.0)
    work["_wr"] = work["_w"] * work["seg_ret"]
    agg = work.groupby(keys, sort=False).agg(_sw=("_w", "sum"), _swr=("_wr", "sum"), _n=("_w", "size")).reset_index()
    work = work.merge(agg, on=keys, how="left")
    # 领先组不包含本股
    work["_sw_ex"] = work["_sw"].to_numpy() - work["_w"].to_numpy()
    work["_swr_ex"] = work["_swr"].to_numpy() - work["_wr"].to_numpy()
    work["_n_ex"] = work["_n"].to_numpy() - (work["_w"].to_numpy() > 0)
    _den = work["_sw_ex"].where(work["_sw_ex"] > 0)
    work["lead_ret"] = (work["_swr_ex"] / _den).where(work["_sw_ex"] > 0)
    work["peer_n"] = work["_n_ex"]

    work = work[work["peer_n"] >= MIN_PEERS]
    gap = (work["lead_ret"] - work["seg_ret"]).astype(float)
    work["signal"] = gap.clip(lower=0.0) if CATCH_ONLY else gap
    work["date"] = pd.to_datetime(work["date"])
    out = work.dropna(subset=["instrument", "signal"])[["date", "instrument", "signal"]]
    out = out.drop_duplicates(["date", "instrument"]).copy()
    out["factor"] = out.groupby("date")["signal"].rank(pct=True) - 0.5
    out["factor"] = out["factor"].fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    # 平台骨架：只保留 8 个约定截面（0930/1000/1030/1100/1130/1330/1400/1430），
    # 并以股票池 (date, instrument) 为全量骨架（信号缺失处填 0），与 485 等已交付因子同口径。
    # 这同时避免了按季度调用时相邻批次重叠行造成的重复键与越界行。
    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)]
    _pool = _pool.drop_duplicates(["date", "instrument"])
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype(str)
    out = out.drop_duplicates(["date", "instrument"])
    out = _pool.merge(out[["date", "instrument", "factor"]], on=["date", "instrument"], how="left")
    out["factor"] = out["factor"].astype("float64").fillna(0.0)
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
