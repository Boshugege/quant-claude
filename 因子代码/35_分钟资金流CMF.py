def main(datasources, start_date, end_date):
    """
    分钟资金流 CMF（31 号的性能优化替代）。

    经济含义：每分钟收盘价在最高最低价区间中的位置 CLV = (2*close - high - low)/(high - low)，
    收在最高价为 +1（买方主导），收在最低价为 -1（卖方主导）；30 分钟内按成交量加权：
      factor = Σ(CLV * volume) / Σ volume
    这是 Chaikin Money Flow 的日内版本，不需要上一分钟数据，因此没有窗口函数。
    只读 close/high/low/volume 4 个字段，是本项目里读数据量最少的因子之一。
    可能有效的理由：盘口因子刻画挂单意图，CMF 刻画已成交资金的方向，信息来源不同；
    小盘股 30 分钟级别的成交方向可能有延续（追涨）或反转（流动性补偿），由评分检验。

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
            CAST(COALESCE(close, 0) AS DOUBLE) AS px,
            CAST(COALESCE(high, 0) AS DOUBLE) AS hi,
            CAST(COALESCE(low, 0) AS DOUBLE) AS lo,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol
        FROM {bar1m}
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
        FROM raw
    ),
    snap AS (
        SELECT
            trading_day, time_segment, instrument_id, vol,
            CASE WHEN hi > lo AND vol > 0 THEN (2.0 * px - hi - lo) / (hi - lo) END AS clv
        FROM seg
        WHERE time_segment > 0
    ),
    win AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(clv * vol) / NULLIF(SUM(CASE WHEN clv IS NOT NULL THEN vol END), 0) AS factor
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
    # 一字板（high == low）整段无成交方向时为缺失；用中性值 0 补齐，避免截面缺失率超限。
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
