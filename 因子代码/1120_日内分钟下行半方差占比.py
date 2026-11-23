def main(datasources, start_date, end_date):
    """【新根因子 方向D：已实现高阶矩】当日截至当前窗口的 1 分钟收益偏度／峰度／下行半方差占比（可滚动 5 日）。
    【方向D变体 1120：日内分钟下行半方差占比】

    假设：彩票偏好与下行风险补偿——日内 1 分钟收益分布的高阶形状（右偏、厚尾、下行波动占比）刻画
    散户博弈与抛压结构，对之后 30 分钟的收益有预测力。

    只用到分钟行情的收价与前收。矩的**分量和**（n, Σr, Σr², Σr³, Σr⁴, Σr⁻²）在 SQL 里按段算出，
    再在 pandas 里按日累加后合成偏度/峰度/下行占比（分量和可加，避免把明细拉进内存）。
    不含反转阶梯、K 线形态、泄漏分量。

    参数：STAT=skew|kurt|dnratio|skew_m_kurt、ROLL_DAYS=0（当日累计）或 5（把当日统计量在股票上做 5 日均值）。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "dnratio"         # skew | kurt | dnratio | skew_m_kurt
    ROLL_DAYS = 0            # 0 = 只用当日累计；5 = 把当日统计量再取过去 5 日均值

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
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
    ),
    cte_ret AS (
        SELECT
            instrument_id, instrument, trading_day, hms, time_segment,
            CASE WHEN px_close > 0 AND pre_close > 0 THEN
                LOG(px_close / COALESCE(NULLIF(LAG(px_close) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms), 0), pre_close))
            ELSE NULL END AS r
        FROM cte_raw
        WHERE time_segment != -1
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
        COUNT(r) AS n,
        SUM(COALESCE(r, 0)) AS s1,
        SUM(COALESCE(r * r, 0)) AS s2,
        SUM(COALESCE(r * r * r, 0)) AS s3,
        SUM(COALESCE(r * r * r * r, 0)) AS s4,
        SUM(CASE WHEN r < 0 THEN r * r ELSE 0 END) AS sdn
    FROM cte_ret
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

    # 当日累计（分量和可加）
    gd = df.groupby(["instrument_id", "trading_day"], sort=False)
    for c in ["n", "s1", "s2", "s3", "s4", "sdn"]:
        df["c_" + c] = gd[c].cumsum()

    n = df["c_n"].astype(float)
    m1 = df["c_s1"] / n
    m2 = df["c_s2"] / n
    m3 = df["c_s3"] / n
    m4 = df["c_s4"] / n
    var = (m2 - m1 * m1).clip(lower=1e-12)
    sd = np.sqrt(var)
    skew = (m3 - 3 * m1 * m2 + 2 * m1 ** 3) / (sd ** 3)
    kurt = (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4) / (var ** 2)
    dnratio = df["c_sdn"] / df["c_s2"].where(df["c_s2"] > 0)

    if STAT == "skew":
        df["stat"] = skew
    elif STAT == "kurt":
        df["stat"] = kurt
    elif STAT == "dnratio":
        df["stat"] = dnratio
    else:
        df["stat"] = skew - kurt
    df["stat"] = pd.Series(df["stat"], index=df.index).replace([np.inf, -np.inf], np.nan)
    df.loc[df["c_n"] < 15, "stat"] = np.nan

    if ROLL_DAYS:
        gg = df.groupby(["instrument_id", "time_segment"], sort=False)
        df["signal"] = gg["stat"].transform(
            lambda x: x.rolling(ROLL_DAYS, min_periods=max(2, ROLL_DAYS - 1)).mean())
    else:
        df["signal"] = df["stat"]

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
