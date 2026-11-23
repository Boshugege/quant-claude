def main(datasources, start_date, end_date):
    """
    价距调整的逐档挂单密度不平衡。

    经济含义：第 i 档买方“密度”= 买 i 量 / (中间价 - 买 i 价)，卖方同理；
    每档比较买卖密度后归一化：z_i = (bv_i * da_i - av_i * db_i) / (bv_i * da_i + av_i * db_i)，
    再对有效档位求平均。单边盘口分钟剔除。
    可能有效的理由：15 号只比较量，忽略了价格空档。卖方档位稀疏（价格跳空大）意味着
    同样的量分散在更宽的价格区间里，向上推动价格的阻力更小。把价格距离纳入后，
    相同挂单量在“密集”和“稀疏”两种盘口下得到不同的压力值，是对 15 号的直接信息增量。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    levels = range(1, 6)

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_price{i}, 0) AS DOUBLE) AS bp{i}" for i in levels]
        + [f"CAST(COALESCE(ask_price{i}, 0) AS DOUBLE) AS ap{i}" for i in levels]
        + [f"CAST(COALESCE(bid_volume{i}, 0) AS DOUBLE) AS bv{i}" for i in levels]
        + [f"CAST(COALESCE(ask_volume{i}, 0) AS DOUBLE) AS av{i}" for i in levels]
    )
    valid = {i: f"(bp{i} > 0 AND ap{i} > 0 AND bp{i} < mid AND ap{i} > mid)" for i in levels}
    dens_terms = " + ".join(
        f"CASE WHEN {valid[i]} THEN (bv{i} * (ap{i} - mid) - av{i} * (mid - bp{i}))"
        f" / (bv{i} * (ap{i} - mid) + av{i} * (mid - bp{i}) + 1e-8) ELSE 0 END"
        for i in levels
    )
    valid_count = " + ".join(f"CASE WHEN {valid[i]} THEN 1 ELSE 0 END" for i in levels)

    sql = f"""
    WITH raw AS (
        SELECT
            instrument_id,
            CAST(CAST(date AS TIMESTAMP) AS DATE) AS trading_day,
            hour(date) * 10000 + minute(date) * 100 + second(date) AS hms,
            {raw_cols}
        FROM {bar1m}
    ),
    seg AS (
        SELECT
            *,
            (bp1 + ap1) / 2.0 AS mid,
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
            CASE WHEN bp1 > 0 AND ap1 > bp1
                 THEN ({dens_terms}) / NULLIF({valid_count}, 0)
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
