def main(datasources, start_date, end_date):
    """【新根因子 第二轮 方向F深挖 1200 起】把"价格在区间里的位置"从当日扩到多日，
    并加上相对今开／昨收的偏离。全部只用**截面时刻及之前**的信息：当日累计用 cummax/cummin/cumsum，
    多日窗口只用**已收盘的整日**高低（前一日及更早），严格因果。不读 vwap 表、不读暴露表。

    参数：STAT=dist_low_m3|dist_low_m5|pos_range_m3|pos_range_m5|dev_open|day_ret。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "dev_open"
    WIN = {"dist_low_m3": 3, "dist_low_m5": 5, "pos_range_m3": 3, "pos_range_m5": 5,
           "dev_open": 1, "day_ret": 1}[STAT]
    bar1m = datasources["bar1m"]
    # 多日窗口要预留足够的暖机天数（5 日窗口 + 非交易日冗余）
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=25)).strftime("%Y-%m-%d %H:%M:%S")
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
    px = df["last_px"].where(df["last_px"] > 0)
    pre = df["pre_close"].where(df["pre_close"] > 0)
    # 今开：当日 09:25 集合竞价结束价（当日第一段的 last_px），只用截面时刻之前的信息
    day_open = df.loc[df["time_segment"] == 93000, ["instrument_id", "trading_day", "last_px"]]
    day_open = day_open.rename(columns={"last_px": "day_open"})
    df = df.merge(day_open, on=["instrument_id", "trading_day"], how="left")
    df["day_open"] = df["day_open"].where(df["day_open"] > 0)

    if STAT in ("dist_low_m3", "dist_low_m5", "pos_range_m3", "pos_range_m5"):
        # 先按 股票×整日 汇总出**已收盘**的整日高低（当日的高只能在收盘后算完，所以当日不用它）
        d = (df.groupby(["instrument_id", "trading_day"], sort=False)
               .agg(day_high=("seg_high", "max"), day_low=("seg_low", "min"))
               .reset_index().sort_values(["instrument_id", "trading_day"]))
        g = d.groupby("instrument_id", sort=False)
        # 前 WIN-1 个**已完成**交易日的高低（shift(1) 把当日排除；rolling 只用过去）
        d["pre_high"] = g["day_high"].transform(
            lambda s: s.shift(1).rolling(WIN - 1, min_periods=1).max())
        d["pre_low"] = g["day_low"].transform(
            lambda s: s.shift(1).rolling(WIN - 1, min_periods=1).min())
        df = df.merge(d[["instrument_id", "trading_day", "pre_high", "pre_low"]],
                      on=["instrument_id", "trading_day"], how="left")
        # 显式赋值：既把类型钉死，也让 preflight 的"列不存在"静态检查能识别这两列
        df["pre_high"] = df["pre_high"].astype("float64")
        df["pre_low"] = df["pre_low"].astype("float64")
        multi_high = np.fmax(df["pre_high"].to_numpy(float), df["cum_high"].to_numpy(float))
        multi_low = np.fmin(df["pre_low"].to_numpy(float), df["cum_low"].to_numpy(float))
        rng = multi_high - multi_low
        if STAT.startswith("dist_low"):
            sig = (px.to_numpy(float) - multi_low) / pre.to_numpy(float)
        else:
            sig = (px.to_numpy(float) - multi_low) / np.where(rng > 0, rng, np.nan)
    elif STAT == "dev_open":
        sig = (px.to_numpy(float) - df["day_open"].to_numpy(float)) / pre.to_numpy(float)
    else:  # day_ret
        sig = (px.to_numpy(float) - pre.to_numpy(float)) / pre.to_numpy(float)

    df["signal"] = pd.Series(sig, index=df.index).replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(["date", "instrument_id"])

    out = df.dropna(subset=["instrument", "signal"])[["date", "instrument", "signal"]]
    out = out.drop_duplicates(["date", "instrument"]).copy()
    out["factor"] = out.groupby("date")["signal"].rank(pct=True) - 0.5
    out["factor"] = out["factor"].fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    # 平台骨架：只保留 8 个约定截面，以股票池 (date, instrument) 为全量骨架（信号缺失处填 0）
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
