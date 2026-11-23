def main(datasources, start_date, end_date):
    """
    委托笔数多视角复合。

    一次 SQL 扫描同时计算 4 个委托笔数不平衡信号：
      f_agg : 五档笔数等权加总后归一化（24 号，公榜 0.40672）
      f_lvl : 逐档笔数归一化后平均（40 号）
      f_far : 远档加权笔数不平衡（41 号，权重同 01 号）
      f_pow : 逐档笔数不平衡的保符号 0.75 次幂平均（21 号幂变换思路）
    每个信号取 30 分钟均值后，在股票池内逐截面转为居中排名，再等权平均。
    可能有效的理由：四个信号都基于目前最强的委托笔数信息、方向一致，
    但聚合方式不同（加总 / 逐档 / 远档 / 非线性），排名合成可抵消各自的聚合噪声。
    不混入挂单量：挂单量 = 笔数 x 平均单量，挂单量因子明显弱于笔数（0.27~0.31 vs 0.41），
    说明平均单量部分基本是噪声，混入会稀释信号。

    稳健性：缺失按 0 计，分钟信号永不为空；最终缺失补 0，每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 10 个字段，单次 GROUP BY，无窗口函数、无 Python UDF。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import math
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    levels = range(1, 6)
    weights = {1: math.exp(-1.2), 2: math.exp(-0.9), 3: math.exp(-0.6), 4: math.exp(-0.3), 5: 1.0}

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_num_orders{i}, 0) AS DOUBLE) AS bn{i}" for i in levels]
        + [f"CAST(COALESCE(ask_num_orders{i}, 0) AS DOUBLE) AS an{i}" for i in levels]
    )
    z = {i: f"((bn{i} - an{i}) / (bn{i} + an{i} + 1e-8))" for i in levels}
    lvl_terms = " + ".join(z[i] for i in levels)
    pow_terms = " + ".join(f"SIGN({z[i]}) * POW(ABS({z[i]}), 0.75)" for i in levels)
    bid_sum = " + ".join(f"bn{i}" for i in levels)
    ask_sum = " + ".join(f"an{i}" for i in levels)
    bid_far = " + ".join(f"bn{i} * {weights[i]:.10f}" for i in levels)
    ask_far = " + ".join(f"an{i} * {weights[i]:.10f}" for i in levels)

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            {raw_cols}
        FROM {bar1m}
    ),
    cte_sums AS (
        SELECT
            *,
            ({bid_sum}) AS sb,
            ({ask_sum}) AS sa,
            ({bid_far}) AS fb,
            ({ask_far}) AS fa
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
            (sb - sa) / (sb + sa + 1e-8) AS s_agg,
            ({lvl_terms}) / 5.0 AS s_lvl,
            (fb - fa) / (fb + fa + 1e-8) AS s_far,
            ({pow_terms}) / 5.0 AS s_pow
        FROM cte_sums
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            AVG(s_agg) AS f_agg,
            AVG(s_lvl) AS f_lvl,
            AVG(s_far) AS f_far,
            AVG(s_pow) AS f_pow
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
        f.f_agg, f.f_lvl, f.f_far, f.f_pow
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    components = ["f_agg", "f_lvl", "f_far", "f_pow"]
    df = df.replace([np.inf, -np.inf], np.nan)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    result = pd.merge(df, stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])

    # 股票池内逐截面居中排名 [-0.5, 0.5]，等权平均；缺失补 0（即截面中位）。
    ranks = result.groupby("date")[components].rank(pct=True) - 0.5
    result["factor"] = ranks.mean(axis=1, skipna=True).astype("float64").fillna(0.0)
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
