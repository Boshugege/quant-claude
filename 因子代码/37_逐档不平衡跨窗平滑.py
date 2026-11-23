def main(datasources, start_date, end_date):
    """
    逐档不平衡跨窗平滑（15 号的 60 分钟版本）。

    经济含义：先按 15 号公式得到每个 30 分钟窗口的逐档不平衡均值 f_t，
    再与同一交易日上一个窗口加权：factor_t = (f_t + 0.5 * f_{t-1}) / 1.5；
    当日第一个窗口没有上一窗口，直接取 f_t。
    可能有效的理由：03 号只用末段 10 分钟（0.218）明显差于用满 30 分钟（0.266），
    说明这个盘口信号是“慢变的状态量”，平均时间越长噪声越小。把上一窗口以半权重纳入，
    相当于有效平均时长约 45 分钟，同时仍以最近 30 分钟为主。只用过去数据，无未来函数。

    性能：分钟级计算全部在 SQL 内向量化完成；跨窗口平滑在 pandas 中对已聚合的
    小表（每股每天 8 行）做 shift，不使用 SQL 窗口函数，内存开销可忽略。
    稳健性：分钟信号永不为空；最终缺失补 0，保证每个截面缺失率为 0。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    PREV_WEIGHT = 0.5

    bar1m = datasources["bar1m"]
    levels = range(1, 6)

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_volume{i}, 0) AS DOUBLE) AS bv{i}" for i in levels]
        + [f"CAST(COALESCE(ask_volume{i}, 0) AS DOUBLE) AS av{i}" for i in levels]
    )
    lvl_terms = " + ".join(f"(bv{i} - av{i}) / (bv{i} + av{i} + 1e-8)" for i in levels)

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
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])

    # 同一交易日内与上一个窗口加权（只用过去数据）。
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    grp = df.groupby("instrument", observed=True, sort=False)
    prev_factor = grp["factor"].shift(1)
    prev_date = grp["date"].shift(1)
    same_day = prev_date.dt.normalize() == df["date"].dt.normalize()
    use_prev = same_day & prev_factor.notna() & df["factor"].notna()
    df["factor"] = np.where(
        use_prev,
        (df["factor"] + PREV_WEIGHT * prev_factor) / (1.0 + PREV_WEIGHT),
        df["factor"],
    )

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    result = pd.merge(df[["date", "instrument", "factor"]], stk_pool, how="inner", on=["date", "instrument"])
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
