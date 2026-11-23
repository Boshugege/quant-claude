def main(datasources, start_date, end_date):
    """
    30 分钟 VWAP 偏离短期反转。

    经济含义：factor = ln(区间成交 VWAP / 区间末中间价)。
    区间末价格跌破本区间成交均价时为正（超跌），涨过成交均价时为负（超涨）。
    区间末价格优先用买一卖一中间价（不受买卖价差反弹影响），单边盘口时退回收盘价。
    可能有效的理由：中证 1000 小盘股日内流动性冲击明显，30 分钟级别的价格偏离
    往往在下一段内部分回吐（流动性提供者获得的补偿）。目标为未来 VWAP，
    与“当前价相对 VWAP 的位置”天然相关。它只用价格和成交额，与所有盘口因子来源不同。

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
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt
        FROM {bar1m}
    ),
    seg AS (
        SELECT
            *,
            CASE WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                 WHEN px_close > 0 THEN px_close
            END AS px,
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
    win AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt) / NULLIF(SUM(vol), 0) AS vwap,
            arg_max(px, hms) AS last_px
        FROM seg
        WHERE time_segment > 0 AND px IS NOT NULL
        GROUP BY trading_day, time_segment, instrument_id
    )
    SELECT w.trading_day, w.time_segment, i.instrument, w.vwap, w.last_px
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
    vwap = df["vwap"].astype("float64")
    last_px = df["last_px"].astype("float64")
    ok = (vwap > 0) & (last_px > 0)
    df["factor"] = np.where(ok, np.log(vwap.where(ok, 1.0) / last_px.where(ok, 1.0)), np.nan)
    df["factor"] = df["factor"].replace([np.inf, -np.inf], np.nan)

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
