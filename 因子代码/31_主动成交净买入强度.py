def main(datasources, start_date, end_date):
    """
    主动成交净买入强度（报价法成交方向判别）。

    经济含义：每分钟用“本分钟成交均价 amount/volume”相对“上一分钟末买一卖一中间价”的位置，
    判断这一分钟的成交主要是主动买还是主动卖：
      s = clip((本分钟成交均价 - 上分钟中间价) / 上分钟半价差, -1, 1)
    s = +1 表示成交基本吃在卖一上（主动买），-1 表示砸在买一上（主动卖）。
    30 分钟内按成交量加权：factor = Σ(s * volume) / Σ volume。
    可能有效的理由：盘口挂单是“意图”，主动成交是“已经付出成本的真实需求”。
    这是 Lee-Ready 报价法在分钟数据上的连续版本，信息来源与静态盘口不同，可与盘口因子互补。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    sql = f"""
    WITH raw AS (
        SELECT
            instrument_id,
            CAST(CAST(date AS TIMESTAMP) AS DATE) AS trading_day,
            hour(date) * 10000 + minute(date) * 100 + second(date) AS hms,
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt
        FROM {bar1m}
    ),
    lagged AS (
        SELECT
            *,
            LAG(bp1) OVER w AS pbp1,
            LAG(ap1) OVER w AS pap1
        FROM raw
        WINDOW w AS (PARTITION BY instrument_id, trading_day ORDER BY hms)
    ),
    seg AS (
        SELECT
            *,
            CASE
                WHEN hms > 90000  AND hms <= 93000  THEN 93000
                WHEN hms > 93000  AND hms <= 100000 THEN 100000
                WHEN hms > 100000 AND hms <= 103000 THEN 103000
                WHEN hms > 103000 AND hms <= 110000 THEN 110000
                WHEN hms > 110000 AND hms <= 113000 THEN 113000
                WHEN hms > 130000 AND hms <= 133000 THEN 133000
                WHEN hms > 133000 AND hms <= 140000 THEN 140000
                WHEN hms > 140000 AND hms <= 143000 THEN 143000
                ELSE -1
            END AS time_segment
        FROM lagged
    ),
    snap AS (
        SELECT
            trading_day, time_segment, instrument_id, vol,
            CASE WHEN vol > 0 AND amt > 0 AND pbp1 > 0 AND pap1 > pbp1
                 THEN GREATEST(-1.0, LEAST(1.0,
                      (amt / vol - (pap1 + pbp1) / 2.0) / ((pap1 - pbp1) / 2.0)))
            END AS side
        FROM seg
        WHERE time_segment > 0
    ),
    win AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(side * vol) / NULLIF(SUM(CASE WHEN side IS NOT NULL THEN vol END), 0) AS factor
        FROM snap
        GROUP BY trading_day, time_segment, instrument_id
    )
    SELECT w.trading_day, w.time_segment, i.instrument, w.factor
    FROM win w
    LEFT JOIN all_instruments i USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()

    seg = df["time_segment"].astype(int)
    df["date"] = (
        pd.to_datetime(df["trading_day"])
        + pd.to_timedelta(seg // 10000, unit="h")
        + pd.to_timedelta(seg // 100 % 100, unit="m")
    )
    df["instrument"] = df["instrument"].astype(str)
    df["factor"] = df["factor"].astype("float64").replace([np.inf, -np.inf], np.nan)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    stk_pool["date"] = pd.to_datetime(stk_pool["date"])
    stk_pool["instrument"] = stk_pool["instrument"].astype(str)

    result = pd.merge(df[["date", "instrument", "factor"]], stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])
    # 当日第一根 bar 没有上一分钟盘口，若某截面只有这一根会整段缺失；用中性值 0 补齐，避免截面缺失率超限。
    result["factor"] = result["factor"].fillna(0.0)
    return result[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


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
