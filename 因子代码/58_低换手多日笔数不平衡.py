def main(datasources, start_date, end_date):
    """
    低换手委托笔数不平衡：41 号窗口信号的跨窗口指数平滑（含跨日）。

    经济含义：先得到每个 30 分钟窗口的远档加权委托笔数不平衡 f_t（41 号，0.41587），
    再按股票沿时间做因果指数加权平均（半衰期 4 个窗口，约半个交易日，跨日连续）。
    探索目的：官方评分中换手率占 20%（换手越低排名越高），IC_IR 和压力场景稳定性也偏好平稳信号。
    37 号仅加入上一窗口半权重就从 0.31033 提升到 0.34280；本因子进一步拉长平滑，
    检验“牺牲少量 IC、换取大幅降低换手和更高稳定性”的取舍。只用过去数据。

    稳健性：缺失按 0 计，分钟信号永不为空；最终缺失补 0，每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 10 个字段，无窗口函数、无 Python UDF；
    跨窗口平滑在 pandas 中对已聚合的小表（每股每天 8 行）完成。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import math
    import numpy as np
    import pandas as pd
    import dai

    HALFLIFE = 4  # 平滑半衰期（窗口数）

    bar1m = datasources["bar1m"]
    # 向前多取若干天数据作为平滑/学习的预热（官方模板说明允许“多取若干天数据作为缓冲”）；
    # 输出只保留 [start_date, end_date] 内的截面（与股票池内连接时自动截断）。
    LOOKBACK_DAYS = 10
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    levels = range(1, 6)
    weights = {1: math.exp(-1.2), 2: math.exp(-0.9), 3: math.exp(-0.6), 4: math.exp(-0.3), 5: 1.0}

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_num_orders{i}, 0) AS DOUBLE) AS bn{i}" for i in levels]
        + [f"CAST(COALESCE(ask_num_orders{i}, 0) AS DOUBLE) AS an{i}" for i in levels]
    )
    bid_sum = " + ".join(f"bn{i} * {weights[i]:.10f}" for i in levels)
    ask_sum = " + ".join(f"an{i} * {weights[i]:.10f}" for i in levels)

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            {raw_cols}
        FROM {bar1m}
    ),
    cte_weighted AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            ({bid_sum}) AS wb,
            ({ask_sum}) AS wa
        FROM cte_raw
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
            (wb - wa) / (wb + wa + 1e-8) AS minute_signal
        FROM cte_weighted
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

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["factor"] = df.groupby("instrument", observed=True, sort=False)["factor"].transform(
        lambda s: s.ewm(halflife=HALFLIFE, min_periods=1, ignore_na=True).mean()
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
