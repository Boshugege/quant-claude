def main(datasources, start_date, end_date):
    """
    三基准反转复合：日内 VWAP 偏离 + 日内区间位置 + 30 分钟反转。

    分量：r_dvwap（67 号，0.77033）、r_rng（61 号，0.65968）、r_30（57 号，0.57571），截面排名等权平均。
    思路：选取目前最强、且参照价格各不相同的三个反转——当日平均成本、当日价格区间、30 分钟前价格。
    64 号（56 + 57）0.77358 已证明不同尺度反转合成的增益很大；本因子用更强的日级别分量替换昨收基准。
    同批 74–76 共用同一段 SQL（在 68–73 的基础上增加最高/最低价）：逐窗口聚合成交额、成交量、窗口末中间价、前收盘价
    （额外聚合 14:30–15:00 窗口，仅用于全天 VWAP 与跨日滚动 VWAP，不进入输出），
    在 pandas 中构造各反转分量：
      r_dvwap : 日内 VWAP 偏离反转 ln(开盘以来累计 VWAP / 当前价)（67 号，0.77033）
      r_day   : 日内反转 ln(昨收 / 当前价)（56 号，0.60243）
      r_30    : 30 分钟反转 ln(上一窗口末价 / 当前价)，当日首个窗口以昨收为基准（57 号，0.57571）
      r_rng   : 日内区间位置反转 (H - P) / (H - L) - 0.5，H/L 为当日至今最高/最低价并纳入昨收（61 号，0.65968）
      r_dvwap_rng : 振幅标准化的日内 VWAP 偏离 (累计 VWAP - P) / (H - L)
    复合时各分量在股票池内逐截面转为居中排名后等权平均（缺失分量不参与）。

    稳健性：最终缺失补 0（中性），每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 8 个字段，单次 GROUP BY，不使用任何窗口函数、不使用 Python UDF；
    跨窗口/跨日运算在 pandas 中对已聚合的小表（每股每天 9 行）完成，只用过去数据。

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
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
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
            px_high,
            px_low,
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
                WHEN hms > '143000' AND hms <= '150000' THEN 150000
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
            MAX(px_high) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low ELSE 1e12 END) AS win_low
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
        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    grp = df.groupby("instrument", observed=True, sort=False)
    grp_day = df.groupby(["instrument", "day"], observed=True, sort=False)
    same_day = grp["day"].shift(1) == df["day"]
    prev_px = grp["last_px"].shift(1).where(same_day, df["pre_close"])
    cum_amt = grp_day["sum_amt"].cumsum()
    cum_vol = grp_day["sum_vol"].cumsum()
    day_vwap = (cum_amt / cum_vol.where(cum_vol > 0)).where(cum_amt > 0)
    df["r_dvwap"] = safe_log_ratio(day_vwap, df["last_px"])
    df["r_day"] = safe_log_ratio(df["pre_close"], df["last_px"])
    df["r_30"] = safe_log_ratio(prev_px, df["last_px"])

    # 当日开盘至今的最高/最低价（纳入昨收），用于区间位置（61 号）与振幅标准化。
    win_high = df["win_high"].where(df["win_high"] > 0)
    win_low = df["win_low"].where(df["win_low"] < 1e11)
    cum_high = win_high.groupby([df["instrument"], df["day"]], observed=True).cummax()
    cum_low = win_low.groupby([df["instrument"], df["day"]], observed=True).cummin()
    H = np.fmax(cum_high, df["pre_close"])
    L = np.fmin(cum_low, df["pre_close"])
    rng = (H - L).where((H - L) > 0)
    df["r_rng"] = (H - df["last_px"]) / rng - 0.5
    df["r_dvwap_rng"] = (day_vwap - df["last_px"]) / rng


    df = df.replace([np.inf, -np.inf], np.nan)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    keep = [c for c in df.columns if c.startswith("r_") or c == "factor"]
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])

    # 股票池内逐截面居中排名后等权平均；缺失分量不参与。
    use = ['r_dvwap', 'r_rng', 'r_30']
    ranks = result.groupby("date")[use].rank(pct=True) - 0.5
    result["factor"] = ranks.mean(axis=1, skipna=True)
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
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
