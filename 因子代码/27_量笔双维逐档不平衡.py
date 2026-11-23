def main(datasources, start_date, end_date):
    """
    量笔双维逐档不平衡。

    经济含义：每一档同时比较“买卖挂单量”和“买卖委托笔数”，两种口径各自逐档归一化，
    再对 5 档 x 2 维共 10 项等权平均；单边盘口（涨跌停封板，买一或卖一价格为 0）的分钟剔除。
    可能有效的理由：15 号已证明逐档归一化最强（0.31）。委托笔数衡量的是“有多少人在排队”，
    不容易被单笔大挂单（可能是撤单/诱单）主导，与挂单量互为验证，等于给每档多了一次独立投票。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    levels = range(1, 6)

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_volume{i}, 0) AS DOUBLE) AS bv{i}" for i in levels]
        + [f"CAST(COALESCE(ask_volume{i}, 0) AS DOUBLE) AS av{i}" for i in levels]
        + [f"CAST(COALESCE(bid_num_orders{i}, 0) AS DOUBLE) AS bn{i}" for i in levels]
        + [f"CAST(COALESCE(ask_num_orders{i}, 0) AS DOUBLE) AS an{i}" for i in levels]
    )
    vol_terms = " + ".join(f"(bv{i} - av{i}) / (bv{i} + av{i} + 1e-8)" for i in levels)
    cnt_terms = " + ".join(f"(bn{i} - an{i}) / (bn{i} + an{i} + 1e-8)" for i in levels)

    sql = f"""
    WITH raw AS (
        SELECT
            instrument_id,
            CAST(CAST(date AS TIMESTAMP) AS DATE) AS trading_day,
            hour(date) * 10000 + minute(date) * 100 + second(date) AS hms,
            COALESCE(bid_price1, 0) AS bp1,
            COALESCE(ask_price1, 0) AS ap1,
            {raw_cols}
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
            trading_day,
            time_segment,
            instrument_id,
            CASE WHEN bp1 > 0 AND ap1 > 0
                 THEN ({vol_terms} + {cnt_terms}) / 10.0
            END AS minute_signal
        FROM seg
        WHERE time_segment > 0
    ),
    win AS (
        SELECT trading_day, time_segment, instrument_id, AVG(minute_signal) AS factor
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
