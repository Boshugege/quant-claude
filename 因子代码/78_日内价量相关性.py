def main(datasources, start_date, end_date):
    """
    日内价量相关性（参考东吴证券金工“高频价量相关性 CPV”，按日内框架改编）。

    原文思路：用分钟价格与分钟成交量的相关系数刻画量价配合；价量正相关（放量上涨、缩量下跌）
    的股票未来收益偏低，原文因子方向为负。
    本因子：开盘至今所有分钟的 corr(价格, 成交量)，价格以昨收为基准归一化（close / pre_close），
    取相反数：价量负相关（放量下跌后、缩量上涨）因子更大。
    信息来源：成交量在价格路径上的分布，与价格反转（只看价格位置）不同。
    同批 77–79 共用同一段 SQL：每分钟计算 bar 内收益 r = ln(close / open)、聪明度权重
    S^2 = r^2 / volume、相对昨收的价格 p = close / pre_close，逐窗口聚合各阶矩与加权成交额，
    在 pandas 中按“开盘至今”累计后构造因子（与 67 号“开盘以来累计 VWAP”的时间框架一致）。

    稳健性：样本不足或方差为 0 时置缺失，最终缺失补 0（中性），每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 6 个字段，单次 GROUP BY，不使用任何窗口函数、不使用 Python UDF；
    累计运算在 pandas 中对已聚合的小表（每股每天 8 行）完成，只用过去数据。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt
        FROM {bar1m}
    ),
    cte_minute AS (
        SELECT
            instrument_id,
            trading_day,
            vol,
            amt,
            CASE WHEN px_open > 0 AND px_close > 0 AND vol > 0 THEN 1.0 ELSE 0.0 END AS ok,
            CASE WHEN px_open > 0 AND px_close > 0 AND vol > 0 THEN LN(px_close / px_open) ELSE 0.0 END AS r,
            CASE WHEN px_close > 0 AND pre_close > 0 AND vol > 0 THEN px_close / pre_close ELSE 0.0 END AS p,
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
            END AS time_segment
        FROM cte_raw
    ),
    cte_weighted AS (
        SELECT
            *,
            CASE WHEN ok > 0 THEN r * r / vol ELSE 0.0 END AS s2
        FROM cte_minute
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt) AS sum_amt,
            SUM(vol) AS sum_vol,
            SUM(s2 * amt) AS s_amt,
            SUM(s2 * vol) AS s_vol,
            SUM(ok) AS n,
            SUM(ok * r) AS r1,
            SUM(ok * r * r) AS r2,
            SUM(ok * r * r * r) AS r3,
            SUM(ok * p) AS p1,
            SUM(ok * p * p) AS p2,
            SUM(ok * vol) AS v1,
            SUM(ok * vol * vol) AS v2,
            SUM(ok * p * vol) AS pv
        FROM cte_weighted
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
        f.sum_amt, f.sum_vol, f.s_amt, f.s_vol, f.n, f.r1, f.r2, f.r3, f.p1, f.p2, f.v1, f.v2, f.pv
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    sums = ["sum_amt", "sum_vol", "s_amt", "s_vol", "n", "r1", "r2", "r3", "p1", "p2", "v1", "v2", "pv"]
    for c in sums:
        df[c] = df[c].astype("float64")
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()
    # 开盘至今累计（同一交易日内按时间累加，只用过去数据）。
    cum = df.groupby(["instrument", "day"], observed=True, sort=False)[sums].cumsum()

    n = cum["n"].where(cum["n"] >= 10)
    cov = cum["pv"] / n - (cum["p1"] / n) * (cum["v1"] / n)
    var_p = cum["p2"] / n - (cum["p1"] / n) ** 2
    var_v = cum["v2"] / n - (cum["v1"] / n) ** 2
    ok = (var_p > 1e-12) & (var_v > 0)
    corr = cov / np.sqrt(var_p.where(ok) * var_v.where(ok))
    df["factor"] = -corr.clip(-1.0, 1.0)

    df["factor"] = df["factor"].replace([np.inf, -np.inf], np.nan)

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
