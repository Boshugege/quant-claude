def main(datasources, start_date, end_date):
    """
    委托笔数远档不平衡 + VWAP 位置 双因子复合。

    两个分量（各自是目前最强的两类、且信息来源完全不同）：
      c_cnt : 远档加权委托笔数不平衡（41 号，公榜 0.41587），30 分钟均值
      c_pos : 当前价相对本窗口成交 VWAP 的位置 ln(窗口末中间价 / 窗口 VWAP)
              （即 39 号 0.40763 的相反数）
    在股票池内逐截面转为居中排名，等权平均。

    方向依据：22 号（微价格 - 中间价）0.19956，26 号（微价格 - 分钟成交 VWAP）0.28578，
    多出来的“中间价 - 成交 VWAP”一项提高了分数，说明“价格高于成交均价”与“买方盘口更强”同号。
    机理推测：若标签为“下一段 VWAP / 本段 VWAP - 1”，可拆成
      ln(下一段VWAP / 当前价) + ln(当前价 / 本段VWAP)，
    第二项在截面时刻已知（c_pos），第一项由盘口排队（c_cnt）预测，两者互补。
    两个分量 IC 相近（约 0.41），截面排名等权即接近最优线性组合。

    稳健性：笔数分钟信号永不为空；某分量缺失时用另一分量，均缺失补 0；每个截面缺失率为 0。
    性能：纯 SQL 向量化，读取 15 个字段，单次 GROUP BY，无窗口函数、无 Python UDF。

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
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,
            {raw_cols}
        FROM {bar1m}
    ),
    cte_minute AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            vol,
            amt,
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
            AVG(s_cnt) AS f_cnt,
            SUM(amt) AS sum_amt,
            SUM(vol) AS sum_vol,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px
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
        f.f_cnt, f.sum_amt, f.sum_vol, f.last_px
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()

    sum_amt = df["sum_amt"].astype("float64")
    sum_vol = df["sum_vol"].astype("float64")
    last_px = df["last_px"].astype("float64")
    ok = (sum_amt > 0) & (sum_vol > 0) & (last_px > 0)
    vwap = (sum_amt / sum_vol.where(ok, 1.0)).where(ok, 1.0)
    df["f_pos"] = np.where(ok, np.log(last_px.where(ok, 1.0) / vwap), np.nan)
    df["f_cnt"] = df["f_cnt"].astype("float64")
    df = df.replace([np.inf, -np.inf], np.nan)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    result = pd.merge(df[["date", "instrument", "f_cnt", "f_pos"]], stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])

    # 股票池内逐截面居中排名 [-0.5, 0.5]，等权平均；缺失分量不参与，均缺失补 0。
    ranks = result.groupby("date")[["f_cnt", "f_pos"]].rank(pct=True) - 0.5
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
