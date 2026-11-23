def main(datasources, start_date, end_date):
    """
    日内累计收益（探索：日级别的价格位置）。

    经济含义：factor = ln(窗口末中间价 / 前收盘价)，即截面时刻相对昨收的日内涨跌幅。
    探索目的：检验日内反转/动量这一更长周期的价格信息，与 30 分钟级别的 39、49 号对比，
    判断标签对哪一时间尺度的价格位置最敏感。
    同批 48–51 共用同一段 SQL（逐窗口成交额、成交量、窗口末中间价、窗口内中间价均值、前收盘价），
    只在最后一步用不同的“价格基准”构造因子；四个分数对比可反推标签的价格结构。

    稳健性：所有价格无效的情形置为缺失，最终缺失补 0（中性），每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 7 个字段，单次 GROUP BY，无窗口函数、无 Python UDF；
    跨窗口运算在 pandas 中对已聚合的小表（每股每天 8 行）完成，只用过去数据。

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
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt
        FROM {bar1m}
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            vol,
            amt,
            pre_close,
            CASE
                WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,
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
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt) AS sum_amt,
            SUM(vol) AS sum_vol,
            SUM(CASE WHEN px > 0 THEN px ELSE 0 END) AS sum_px,
            SUM(CASE WHEN px > 0 THEN 1 ELSE 0 END) AS n_px,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close
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
        f.sum_amt, f.sum_vol, f.sum_px, f.n_px, f.last_px, f.pre_close
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["sum_amt", "sum_vol", "sum_px", "n_px", "last_px", "pre_close"]:
        df[c] = df[c].astype("float64")

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    df["vwap"] = (df["sum_amt"] / df["sum_vol"].where(df["sum_vol"] > 0)).where(df["sum_amt"] > 0)
    df["twap"] = (df["sum_px"] / df["n_px"].where(df["n_px"] > 0)).where(df["sum_px"] > 0)
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)

    df["factor"] = safe_log_ratio(df["last_px"], df["pre_close"])

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
