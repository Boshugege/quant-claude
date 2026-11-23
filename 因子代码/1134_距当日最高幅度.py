def main(datasources, start_date, end_date):
    """【新根因子 方向F：因果日内价格位置】当日截至当前窗口的相对位置与对当日累计 VWAP 的偏离。
    【方向F变体 1134：距当日最高幅度】

    假设：日内相对位置（现价在当日已走出的高低区间里的位置）与"相对当日累计成交均价 VWAP 的偏离"
    刻画当天的买卖压力与套牢/浮盈结构，对之后 30 分钟收益有预测力。

    全部**只用当日截至当前窗口**的信息：SQL 按段给末价、段内最高/最低、段内成交额与成交量，
    pandas 里做当日累计最高/最低/VWAP（cummax/cummin/cumsum，严格因果）。不读 vwap 表。
    不含反转阶梯、K 线形态、泄漏分量。

    参数：STAT=pos_range|dev_vwap|dev_vwap_norm|dist_low|dist_high|blend。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "dist_high"
    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(b.high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(b.low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS amt,
            CAST(COALESCE(b.volume, 0) AS DOUBLE) AS vol,
            CASE
                WHEN strftime(b.date, '%H%M%S') > '090000' AND strftime(b.date, '%H%M%S') <= '093000' THEN 93000
                WHEN strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '100000' THEN 100000
                WHEN strftime(b.date, '%H%M%S') > '100000' AND strftime(b.date, '%H%M%S') <= '103000' THEN 103000
                WHEN strftime(b.date, '%H%M%S') > '103000' AND strftime(b.date, '%H%M%S') <= '110000' THEN 110000
                WHEN strftime(b.date, '%H%M%S') > '110000' AND strftime(b.date, '%H%M%S') <= '113000' THEN 113000
                WHEN strftime(b.date, '%H%M%S') > '130000' AND strftime(b.date, '%H%M%S') <= '133000' THEN 133000
                WHEN strftime(b.date, '%H%M%S') > '133000' AND strftime(b.date, '%H%M%S') <= '140000' THEN 140000
                WHEN strftime(b.date, '%H%M%S') > '140000' AND strftime(b.date, '%H%M%S') <= '143000' THEN 143000
                WHEN strftime(b.date, '%H%M%S') > '143000' AND strftime(b.date, '%H%M%S') <= '150000' THEN 150000
                ELSE -1
            END AS time_segment
        FROM {bar1m} b
        LEFT JOIN all_instruments ai USING (instrument_id)
    )
    SELECT
        CAST(CONCAT(
            trading_day, ' ',
            strftime(strptime(LPAD(CAST(time_segment AS VARCHAR), 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        trading_day,
        time_segment,
        instrument_id,
        MAX(instrument) AS instrument,
        MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                  OR (time_segment = 93000 AND hms = '092500') THEN px_close ELSE 0 END) AS last_px,
        MAX(px_high) AS seg_high,
        MIN(CASE WHEN px_low > 0 THEN px_low END) AS seg_low,
        SUM(amt) AS seg_amt,
        SUM(vol) AS seg_vol,
        MAX(pre_close) AS pre_close
    FROM cte_raw
    WHERE time_segment != -1
    GROUP BY trading_day, time_segment, instrument_id
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    if not len(df):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    df["instrument"] = df["instrument"].astype(str)
    df["instrument_id"] = df["instrument_id"].astype(str)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument_id", "trading_day", "time_segment"]).reset_index(drop=True)

    keys = ["instrument_id", "trading_day"]
    df["cum_high"] = df.groupby(keys, sort=False)["seg_high"].cummax()
    df["cum_low"] = df.groupby(keys, sort=False)["seg_low"].cummin()
    df["cum_amt"] = df.groupby(keys, sort=False)["seg_amt"].cumsum()
    df["cum_vol"] = df.groupby(keys, sort=False)["seg_vol"].cumsum()
    px = df["last_px"].where(df["last_px"] > 0)
    pre = df["pre_close"].where(df["pre_close"] > 0)
    vwap = (df["cum_amt"] / df["cum_vol"].where(df["cum_vol"] > 0)).where(df["cum_amt"] > 0)
    rng = (df["cum_high"] - df["cum_low"])
    pos_range = (px - df["cum_low"]) / rng.where(rng > 0)
    dev_vwap = (px - vwap) / pre
    dev_norm = (px - vwap) / rng.where(rng > 0)

    if STAT == "pos_range":
        df["signal"] = pos_range
    elif STAT == "dev_vwap":
        df["signal"] = dev_vwap
    elif STAT == "dev_vwap_norm":
        df["signal"] = dev_norm
    elif STAT == "dist_low":
        df["signal"] = (px - df["cum_low"]) / pre
    elif STAT == "dist_high":
        df["signal"] = (px - df["cum_high"]) / pre
    else:
        df["signal"] = np.nan
        df["_p"] = pos_range
        df["_d"] = dev_vwap
    df["signal"] = pd.Series(df["signal"], index=df.index).replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(["date", "instrument_id"])

    out = df.dropna(subset=["instrument", "signal"])[["date", "instrument", "signal"]]
    out = out.drop_duplicates(["date", "instrument"]).copy()
    if STAT == "blend":
        r1 = df.groupby("date")["_p"].rank(pct=True) - 0.5
        r2 = df.groupby("date")["_d"].rank(pct=True) - 0.5
        out = df[["date", "instrument"]].copy()
        out["factor"] = (r1.to_numpy() + r2.to_numpy()) / 2.0
    else:
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
