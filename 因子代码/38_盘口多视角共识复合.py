def main(datasources, start_date, end_date):
    """
    盘口多视角共识复合（只用挂单量与挂单价格，不用委托笔数）。

    一次 SQL 扫描同时计算 4 个已被公榜验证或直接派生自最强因子的盘口信号：
      c_lvl  : 逐档不平衡平均（15 号，0.31033）
      c_pow  : 逐档幂变换不平衡（15 x 21 结合）
      c_tail : 外侧两档合计不平衡的 0.75 次幂（21 号，0.29363）
      c_coh  : 买四/买五一致性门控（16 号，0.29890）
      c_dens : 价距调整的逐档密度不平衡（考虑价格空档，信息增量）
    每个信号取 30 分钟均值后，在股票池内逐截面转为居中排名，再等权平均。
    可能有效的理由：这些信号方向一致（买方挂单更强 -> 正），噪声来源不同
    （全档 vs 外档、线性 vs 非线性、量 vs 价距密度），排名等权合成可抵消各自噪声。

    稳健性：所有分钟信号永不为空（无效档位按 0 计），最终缺失补 0，每个截面缺失率为 0。
    性能：纯 SQL 向量化 + 单次 GROUP BY，读取 20 个字段，无窗口函数。

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
    z = {i: f"((bv{i} - av{i}) / (bv{i} + av{i} + 1e-8))" for i in levels}
    lvl_terms = " + ".join(z[i] for i in levels)
    pow_terms = " + ".join(f"SIGN({z[i]}) * POW(ABS({z[i]}), 0.75)" for i in levels)
    tail = "((bv4 + bv5 - av4 - av5) / (bv4 + bv5 + av4 + av5 + 1e-8))"
    valid = {i: f"(bp1 > 0 AND ap1 > bp1 AND bp{i} > 0 AND ap{i} > 0 AND bp{i} < mid AND ap{i} > mid)" for i in levels}
    dens_terms = " + ".join(
        f"CASE WHEN {valid[i]} THEN (bv{i} * (ap{i} - mid) - av{i} * (mid - bp{i}))"
        f" / (bv{i} * (ap{i} - mid) + av{i} * (mid - bp{i}) + 1e-8) ELSE 0 END"
        for i in levels
    )
    valid_count = " + ".join(f"CASE WHEN {valid[i]} THEN 1 ELSE 0 END" for i in levels)

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            {raw_cols}
        FROM {bar1m}
    ),
    cte_mid AS (
        SELECT *, (bp1 + ap1) / 2.0 AS mid
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
            ({lvl_terms}) / 5.0 AS s_lvl,
            ({pow_terms}) / 5.0 AS s_pow,
            SIGN({tail}) * POW(ABS({tail}), 0.75) AS s_tail,
            0.5 * ({z[4]} + {z[5]}) * (1.0 - 0.5 * ABS({z[4]} - {z[5]})) AS s_coh,
            ({dens_terms}) / (CASE WHEN ({valid_count}) > 0 THEN ({valid_count}) ELSE 1 END) AS s_dens
        FROM cte_mid
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            AVG(s_lvl) AS f_lvl,
            AVG(s_pow) AS f_pow,
            AVG(s_tail) AS f_tail,
            AVG(s_coh) AS f_coh,
            AVG(s_dens) AS f_dens
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
        f.f_lvl, f.f_pow, f.f_tail, f.f_coh, f.f_dens
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    components = ["f_lvl", "f_pow", "f_tail", "f_coh", "f_dens"]
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
