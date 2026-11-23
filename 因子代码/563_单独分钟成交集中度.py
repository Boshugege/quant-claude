def main(datasources, start_date, end_date):
    """窗口内分钟成交额集中度（HHI）的 40 窗口均值，取负：分散成交偏反转（09:30 置零版）。

    分钟级微观结构新轴，只用 1 分钟行情；不带任何头部叠加、不带低通。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    LOOKBACK_DAYS = 400
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
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
            pre_close,
            px_high,
            px_low,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN amt ELSE 0
            END AS amt_c,
            CASE WHEN px_close > 0 THEN px_close ELSE 0 END AS px,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high = px_low THEN 1 ELSE 0 END AS flat_c,
            CASE WHEN vol > 0 THEN 1 ELSE 0 END AS trad_c,
            CASE
                WHEN hms > '090000' AND hms <= '093000' THEN 93000
                WHEN hms > '093000' AND hms <= '100000' THEN 100000
                WHEN hms > '100000' AND hms <= '103000' THEN 103000
                WHEN hms > '103000' AND hms <= '110000' THEN 110000
                WHEN hms > '110000' AND hms <= '113000' THEN 113000
                WHEN hms > '130000' AND hms <= '133000' THEN 133000
                WHEN hms > '133000' AND hms <= '140000' THEN 140000
                WHEN hms > '140000' AND hms <= '143000' THEN 143000
                WHEN hms > '143000' AND hms <= '150000' THEN 150000
                ELSE -1
            END AS time_segment
        FROM cte_raw
    ),
    cte_seq AS (
        SELECT
            instrument_id, trading_day, hms, pre_close, px_high, px_low, amt_c, px, flat_c, trad_c, time_segment,
            LAG(px) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS px_l1,
            LAG(px, 2) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS px_l2,
            LAG(amt_c) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS amt_l1
        FROM cte_snapshot
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt_c) AS sum_amt,
            SUM(amt_c * amt_c) AS sum_amt2,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND (px - px_l1) <> 0 AND (amt_c - amt_l1) <> 0 AND (px - px_l1) * (amt_c - amt_l1) > 0 THEN 1 ELSE 0 END) AS n_vpsync,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND (px - px_l1) <> 0 AND (amt_c - amt_l1) <> 0 THEN 1 ELSE 0 END) AS n_vppair,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 AND (px - px_l1) <> 0 AND (px_l1 - px_l2) <> 0 AND (px - px_l1) * (px_l1 - px_l2) > 0 THEN 1 ELSE 0 END) AS n_ppsame,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 AND (px - px_l1) <> 0 AND (px_l1 - px_l2) <> 0 THEN 1 ELSE 0 END) AS n_pppair,
            SUM(flat_c) AS n_flat,
            SUM(trad_c) AS n_trad,
            COUNT(*) AS n_min,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low
        FROM cte_seq
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
        f.sum_amt, f.sum_amt2, f.n_vpsync, f.n_vppair, f.n_ppsame, f.n_pppair, f.n_flat, f.n_trad, f.n_min,
        f.last_px, f.pre_close, f.win_high, f.win_low
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_amt2", "n_vpsync", "n_vppair", "n_ppsame", "n_pppair", "n_flat", "n_trad", "n_min",
              "last_px", "pre_close", "win_high", "win_low"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    pool_hist = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [query_start, end_date]},
        compression=True,
    ).df()
    pool_hist["date"] = pd.to_datetime(pool_hist["date"])
    pool_hist["instrument"] = pool_hist["instrument"].astype(str)
    stk_pool = pool_hist[(pool_hist["date"] >= pd.to_datetime(start_date)) & (pool_hist["date"] <= pd.to_datetime(end_date))]
    universe = set(pool_hist["instrument"].unique())
    df = df[df["instrument"].isin(universe)]
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()
    df = df.replace([np.inf, -np.inf], np.nan)

    # 窗口内分钟结构指标
    _amt = df["sum_amt"].where(df["sum_amt"] > 0)
    _hhi = df["sum_amt2"] / (_amt * _amt)
    _sync = df["n_vpsync"] / df["n_vppair"].where(df["n_vppair"] > 0)
    _pp = df["n_ppsame"] / df["n_pppair"].where(df["n_pppair"] > 0)
    _flat = df["n_flat"] / df["n_min"].where(df["n_min"] > 0)
    _halt = 1.0 - df["n_trad"] / df["n_min"].where(df["n_min"] > 0)

    def _smooth(s):
        return s.groupby(df["instrument"], sort=False).transform(
            lambda x: x.rolling(40, min_periods=10).mean())

    def _rk(col, sign):
        return sign * (col.groupby(df["date"]).rank(pct=True) - 0.5)

    df["c_hhi"] = _rk(_smooth(_hhi), -1.0)
    keep = ["c_hhi"]
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool[["date", "instrument"]], how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)
    result["signal"] = result["c_hhi"]
    result["factor"] = result["signal"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面。
    result.loc[result["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
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
