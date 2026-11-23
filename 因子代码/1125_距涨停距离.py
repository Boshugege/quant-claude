def main(datasources, start_date, end_date):
    """【新根因子 方向E：价格约束】当日截至当前窗口与涨跌停板的距离／触板与开板情况。
    【方向E变体 1125：距涨停距离】

    假设：临近涨跌停时流动性枯竭与磁吸效应会让价格继续朝板的方向走；触板后开板则说明抛压/承接被消化，
    之后的方向与"贴板时间"和"开板次数"有关。对之后 30 分钟收益有预测力。

    只用分钟行情的 high/low/close/pre_close。涨跌停比例按板块判定（科创板 688／创业板 30 为 20%，
    北交所 4/8 开头为 30%，其余 10%；ST 的 5% 没有信息，按 10% 处理并记入局限）。
    不含反转阶梯、K 线形态、泄漏分量。

    参数：STAT=pos_band|dist_up|dist_min|touch_up|open_up|dist_x_touch。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "dist_up"

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
        MAX(pre_close) AS pre_close,
        SUM(CASE WHEN px_high > 0 AND px_low > 0 THEN 1 ELSE 0 END) AS n_min
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

    code = df["instrument"].str.split(".").str[0].fillna("")
    limit_pct = pd.Series(0.10, index=df.index)
    limit_pct = limit_pct.mask(code.str.startswith("688") | code.str.startswith("30"), 0.20)
    limit_pct = limit_pct.mask(code.str.startswith("4") | code.str.startswith("8"), 0.30)
    pre = df["pre_close"].where(df["pre_close"] > 0)
    up_limit = pre * (1.0 + limit_pct)
    dn_limit = pre * (1.0 - limit_pct)
    px = df["last_px"].where(df["last_px"] > 0)

    df["touch_up_seg"] = ((df["seg_high"] >= up_limit) & (pre > 0)).astype(float)
    df["touch_dn_seg"] = ((df["seg_low"] > 0) & (df["seg_low"] <= dn_limit) & (pre > 0)).astype(float)
    df["open_up_seg"] = df["touch_up_seg"] * ((px < up_limit) | (px.isna())).astype(float)

    keys = ["instrument_id", "trading_day"]
    df["cum_touch_up"] = df.groupby(keys, sort=False)["touch_up_seg"].cumsum()
    df["cum_touch_dn"] = df.groupby(keys, sort=False)["touch_dn_seg"].cumsum()
    df["cum_open_up"] = df.groupby(keys, sort=False)["open_up_seg"].cumsum()
    df["cum_min"] = df.groupby(keys, sort=False)["n_min"].cumsum()

    band = (up_limit - dn_limit).where(pre > 0)
    pos = (px - dn_limit) / band.where(band > 0)
    dist_up = (up_limit - px) / pre
    dist_dn = (px - dn_limit) / pre
    dist_min = pd.concat([dist_up, dist_dn], axis=1).min(axis=1)

    if STAT == "pos_band":
        df["signal"] = pos
    elif STAT == "dist_up":
        df["signal"] = dist_up
    elif STAT == "dist_min":
        df["signal"] = dist_min
    elif STAT == "touch_up":
        df["signal"] = df["cum_touch_up"] / df["cum_min"].where(df["cum_min"] > 0)
    elif STAT == "open_up":
        df["signal"] = df["cum_open_up"] / df["cum_min"].where(df["cum_min"] > 0)
    else:
        df["signal"] = dist_up * (1.0 + df["cum_touch_up"] / df["cum_min"].where(df["cum_min"] > 0))
    df["signal"] = pd.Series(df["signal"], index=df.index).replace([np.inf, -np.inf], np.nan)

    out = df.dropna(subset=["instrument", "signal"])[["date", "instrument", "signal"]]
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
