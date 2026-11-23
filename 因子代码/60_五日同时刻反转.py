def main(datasources, start_date, end_date):
    """
    五日同时刻反转：factor = ln(5 个交易日前同一截面时刻的复权价 / 当前复权价)。

    经济含义：过去一周跌得越多，因子越大。复权价 = 中间价 x 复权因子，消除除权除息跳空；
    与 5 天前“同一时刻”比较，避开日内季节性（开盘、午盘的系统性价格模式）。
    探索目的：56 号（日内反转，0.60243）、57 号（30 分钟反转，0.57571）都证明反转为正；
    本因子检验周度尺度的反转（A 股经典的周反转效应）。周收益变化很慢，换手远低于日内信号，
    有利于占 20% 的换手率排名。需要向前多取 15 个自然日数据；取不到时退回日内反转（56 号）。
    同批 60–63 共用同一段 SQL：逐窗口聚合成交额、成交量、窗口末中间价、前收盘价、复权因子、
    窗口最高/最低价、当日开盘价、分钟收盘位置资金流；只在最后一步构造不同的反转信号。

    稳健性：价格无效的情形置为缺失，最终缺失补 0（中性），每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 10 个字段，单次 GROUP BY，不使用任何窗口函数（LAG 等）、
    不使用 Python UDF；跨窗口/跨日运算在 pandas 中对已聚合的小表（每股每天 8 行）完成，只用过去数据。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取若干天数据用于计算多日收益（官方模板说明允许“多取若干天数据作为缓冲”）；
    # 输出只保留 [start_date, end_date] 内的截面（与股票池内连接时自动截断）。
    LOOKBACK_DAYS = 15
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(adjust_factor, 0) AS DOUBLE) AS adj,
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
            adj,
            px_high,
            px_low,
            CASE WHEN hms > '092500' AND hms <= '093100' THEN px_open ELSE 0 END AS open_part,
            CASE
                WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,
            CASE WHEN px_high > px_low AND px_low > 0 AND vol > 0
                 THEN vol * (2.0 * px_close - px_high - px_low) / (px_high - px_low)
                 ELSE 0
            END AS clv_vol,
            CASE WHEN px_high > px_low AND px_low > 0 AND vol > 0 THEN vol ELSE 0 END AS clv_den,
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
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(adj) AS adj,
            MAX(px_high) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low ELSE 1e12 END) AS win_low,
            MAX(open_part) AS open_part,
            SUM(clv_vol) AS clv_vol,
            SUM(clv_den) AS clv_den
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
        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.adj,
        f.win_high, f.win_low, f.open_part, f.clv_vol, f.clv_den
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    num_cols = ["sum_amt", "sum_vol", "last_px", "pre_close", "adj", "win_high", "win_low",
                "open_part", "clv_vol", "clv_den"]
    for c in num_cols:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    K_DAYS = 5
    df["seg"] = df["date"].dt.hour * 10000 + df["date"].dt.minute * 100
    df["adj_px"] = (df["last_px"] * df["adj"]).where(df["adj"] > 0)
    ref = df.groupby(["instrument", "seg"], observed=True, sort=False)["adj_px"].shift(K_DAYS)
    df["factor"] = safe_log_ratio(ref, df["adj_px"])
    # 兜底：评估期开头若取不到 5 日前数据（平台表不含更早数据时），退回以昨收为基准的日内反转，
    # 避免整个截面为常数。
    df["factor"] = df["factor"].fillna(safe_log_ratio(df["pre_close"], df["last_px"]))

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
