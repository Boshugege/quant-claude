def main(datasources, start_date, end_date):
    """
    逐档委托笔数不平衡（24 号的逐档归一化版本）。

    经济含义：每档先计算委托笔数不平衡 (nb_i - na_i)/(nb_i + na_i)，再五档等权平均，
    30 分钟内取均值。
    可能有效的理由：24 号（五档笔数加总后归一化）公榜 0.40672，是目前最强；
    挂单量上“加总 -> 逐档”曾把 02 号 0.26556 提升到 15 号 0.31033。
    逐档归一化让每一档的排队方向等权投票，不被某一档的大量排队支配。

    稳健性：缺失按 0 计，分钟信号永不为空；最终缺失补 0，每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 10 个字段，无窗口函数、无 Python UDF。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    levels = range(1, 6)

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_num_orders{i}, 0) AS DOUBLE) AS bn{i}" for i in levels]
        + [f"CAST(COALESCE(ask_num_orders{i}, 0) AS DOUBLE) AS an{i}" for i in levels]
    )
    lvl_terms = " + ".join(f"(bn{i} - an{i}) / (bn{i} + an{i} + 1e-8)" for i in levels)

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            {raw_cols}
        FROM {bar1m}
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            CASE
                WHEN hms > '090000' AND hms <= '093000' THEN 93000
                WHEN hms > '093000' AND hms <= '100000' THEN 100000
                WHEN hms > '100000' AND hms <= '103000' THEN 103000
                WHEN hms > '103000' AND hms <= '110000' THEN 110000
                WHEN hms > '110000' AND hms <= '113000' THEN 113000
                WHEN hms > '130000' AND hms <= '133000' THEN 133000
                WHEN hms > '133000' AND hms <= '140000' THEN 140000
                WHEN hms > '140000' AND hms <= '143000' THEN 143000
                ELSE -1
            END AS time_segment,
            ({lvl_terms}) / 5.0 AS minute_signal
        FROM cte_raw
    ),
    cte_window AS (
        SELECT trading_day, time_segment, instrument_id, AVG(minute_signal) AS factor
        FROM cte_snapshot
        WHERE time_segment != -1
        GROUP BY instrument_id, trading_day, time_segment
    )
    SELECT
        CAST(CONCAT(
            f.trading_day,
            ' ',
            strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        all_instruments.instrument,
        f.factor AS factor
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()

    df = df.replace([np.inf, -np.inf], np.nan)
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    result = pd.merge(df, stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])
    result["factor"] = result["factor"].astype("float64").fillna(0.0)
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


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
