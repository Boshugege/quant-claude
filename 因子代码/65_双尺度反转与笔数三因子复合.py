def main(datasources, start_date, end_date):
    """
    日内反转 + 30 分钟反转 + 远档委托笔数不平衡 三因子复合。

    分量：r_day（56 号，0.60243）、r_30（57 号，0.57571）、cnt（41 号，0.41587），截面排名等权平均。
    思路：价格反转刻画“已经发生的超跌”，委托笔数刻画“当前排队的买卖意愿”，来源完全不同；
    47 号（笔数 + VWAP 反转）0.46613 已高于两者单独的分数，本因子把反转部分换成更强的两个价格反转。
    同批 64–67 共用同一段 SQL：逐窗口聚合远档加权委托笔数不平衡（41 号）、成交额、成交量、
    窗口末中间价、前收盘价；在 pandas 中构造各分量：
      r_day  : 日内反转 ln(昨收 / 当前价)（56 号，0.60243）
      r_30   : 30 分钟反转 ln(上一窗口末价 / 当前价)，当日首个窗口以昨收为基准（57 号，0.57571）
      cnt    : 远档加权委托笔数不平衡（41 号，0.41587）
      r_dvwap: 日内 VWAP 偏离反转 ln(开盘以来累计 VWAP / 当前价)
    复合时各分量在股票池内逐截面转为居中排名后按权重平均（缺失分量不参与）。

    稳健性：最终缺失补 0（中性），每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 16 个字段，单次 GROUP BY，不使用任何窗口函数、不使用 Python UDF；
    跨窗口运算在 pandas 中对已聚合的小表（每股每天 8 行）完成，只用过去数据。

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

    count_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_num_orders{i}, 0) AS DOUBLE) AS bn{i}" for i in levels]
        + [f"CAST(COALESCE(ask_num_orders{i}, 0) AS DOUBLE) AS an{i}" for i in levels]
    )
    bid_far = " + ".join(f"bn{i} * {weights[i]:.10f}" for i in levels)
    ask_far = " + ".join(f"an{i} * {weights[i]:.10f}" for i in levels)

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
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,
            {count_cols}
        FROM {bar1m}
    ),
    cte_minute AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            vol,
            amt,
            pre_close,
            ({bid_far}) AS fb,
            ({ask_far}) AS fa,
            CASE
                WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px
        FROM cte_raw
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            vol,
            amt,
            pre_close,
            px,
            (fb - fa) / (fb + fa + 1e-8) AS s_cnt,
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
        FROM cte_minute
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            AVG(s_cnt) AS cnt,
            SUM(amt) AS sum_amt,
            SUM(vol) AS sum_vol,
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
        f.cnt, f.sum_amt, f.sum_vol, f.last_px, f.pre_close
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["cnt", "sum_amt", "sum_vol", "last_px", "pre_close"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    grp = df.groupby("instrument", observed=True, sort=False)
    same_day = grp["day"].shift(1) == df["day"]
    prev_px = grp["last_px"].shift(1).where(same_day, df["pre_close"])
    df["r_day"] = safe_log_ratio(df["pre_close"], df["last_px"])
    df["r_30"] = safe_log_ratio(prev_px, df["last_px"])
    grp_day = df.groupby(["instrument", "day"], observed=True, sort=False)
    cum_amt = grp_day["sum_amt"].cumsum()
    cum_vol = grp_day["sum_vol"].cumsum()
    day_vwap = (cum_amt / cum_vol.where(cum_vol > 0)).where(cum_amt > 0)
    df["r_dvwap"] = safe_log_ratio(day_vwap, df["last_px"])
    df = df.replace([np.inf, -np.inf], np.nan)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    comps = ["r_day", "r_30", "cnt", "r_dvwap"]
    result = pd.merge(df[["date", "instrument"] + comps], stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])

    # 股票池内逐截面居中排名后按权重平均；缺失分量不参与加权。
    WEIGHTS = {'r_day': 1.0, 'r_30': 1.0, 'cnt': 1.0}
    use = list(WEIGHTS)
    ranks = result.groupby("date")[use].rank(pct=True) - 0.5
    vals = ranks.to_numpy(dtype="float64")
    w = np.array([WEIGHTS[c] for c in use], dtype="float64")
    has = ~np.isnan(vals)
    den = (has * w).sum(axis=1)
    num = np.nansum(np.nan_to_num(vals) * w, axis=1)
    result["factor"] = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)

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
