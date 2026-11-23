def main(datasources, start_date, end_date):
    """【新根因子 方向G：价格冲击与流动性】1 分钟 Amihud（|r|÷成交额）与 Kyle λ（带号成交额对收益的斜率）。
    【方向G变体 1139：当日累计Kyle斜率】

    假设：单位成交额推动价格的能力（冲击成本）刻画流动性与知情交易比例；冲击成本高的股票，
    之后的收益补偿结构不同，对之后 30 分钟收益有预测力。

    只用分钟行情的 close/amount：带号用 tick rule（本分钟收益符号）。收益与带号成交额的
    分量和（n, Σ|r|/amt, Σr, Σs, Σrs, Σs²）在 SQL 里按段算，回归斜率在 pandas 里合成（分量和可加）。
    不含反转阶梯、K 线形态、泄漏分量。

    参数：STAT=amihud|kyle、WINDOW=seg|day（当段 / 当日累计）、ROLL_DAYS=0|5（再取过去 5 日均值）。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "kyle"
    WINDOW = "day"
    ROLL_DAYS = 0

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument AS instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS amt,
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
    cte_min AS (
        SELECT
            instrument_id, instrument, trading_day, hms, time_segment, amt,
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
        SUM(CASE WHEN r IS NOT NULL AND amt > 0 THEN ABS(r) / amt ELSE 0 END) AS s_amihud,
        SUM(COALESCE(r, 0)) AS s_r,
        SUM(CASE WHEN r IS NOT NULL THEN SIGN(r) * amt ELSE 0 END) AS s_s,
        SUM(CASE WHEN r IS NOT NULL THEN r * SIGN(r) * amt ELSE 0 END) AS s_rs,
        SUM(CASE WHEN r IS NOT NULL THEN amt * amt ELSE 0 END) AS s_ss
    FROM cte_min
    GROUP BY trading_day, time_segment, instrument_id
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    if not len(df):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    df["instrument_id"] = df["instrument_id"].astype(str)
    df["instrument"] = df["instrument"].astype(str)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument_id", "trading_day", "time_segment"]).reset_index(drop=True)

    keys = ["instrument_id", "trading_day"]
    if WINDOW == "day":
        for c in ["n", "s_amihud", "s_r", "s_s", "s_rs", "s_ss"]:
            df["c_" + c] = df.groupby(keys, sort=False)[c].cumsum()
    else:
        for c in ["n", "s_amihud", "s_r", "s_s", "s_rs", "s_ss"]:
            df["c_" + c] = df[c]

    n = df["c_n"].astype(float)
    amihud = df["c_s_amihud"] / n.where(n > 0) * 1e8
    vr = df["c_s_ss"] - df["c_s_s"] ** 2 / n
    cov = df["c_s_rs"] - df["c_s_r"] * df["c_s_s"] / n
    kyle = cov / vr.where(vr > 0)

    if STAT == "amihud":
        df["stat"] = amihud
    else:
        df["stat"] = kyle
    df["stat"] = pd.Series(df["stat"], index=df.index).replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["instrument", "instrument_id"])
    df.loc[df["c_n"] < 5, "stat"] = np.nan

    if ROLL_DAYS:
        df["signal"] = df.groupby(["instrument_id", "time_segment"], sort=False)["stat"].transform(
            lambda x: x.rolling(ROLL_DAYS, min_periods=max(2, ROLL_DAYS - 1)).mean())
    else:
        df["signal"] = df["stat"]

    out = df.dropna(subset=["signal"])[["date", "instrument", "signal"]]
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
