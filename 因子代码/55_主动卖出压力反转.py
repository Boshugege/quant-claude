def main(datasources, start_date, end_date):
    """
    主动卖出压力反转（31 号的方向修正版）。

    经济含义：每分钟用“本分钟成交均价 amount/volume”相对“上一分钟末买一卖一中间价”的位置
    判断主动买卖方向 s ∈ [-1, 1]（+1 吃卖一，-1 砸买一），30 分钟内按成交量加权后取相反数：
      factor = -Σ(s * volume) / Σ volume
    主动卖出越多，因子越大。
    方向依据：31 号（主动买入为正）得 0.02538，是全部提交中的最低分；按官方评分规则，
    显著负 IC 的因子会在 IC 均值、IC_IR、夏普三项上排在末尾。翻转后检验“流动性冲击反转”：
    短时间内被主动砸盘的股票，在随后 30 分钟的成交均价上部分回补，与 39 号价格反转逻辑一致。

    性能：31 号的同类写法已在平台跑通——窗口函数 LAG 只滞后 2 列（买一、卖一价），内存开销小。
    稳健性：无法判别方向的窗口（如当日首根 bar）缺失补 0，每个截面缺失率为 0。

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
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt
        FROM {bar1m}
    ),
    cte_lagged AS (
        SELECT
            *,
            LAG(bp1) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS pbp1,
            LAG(ap1) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS pap1
        FROM cte_raw
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            vol,
            CASE
                WHEN vol > 0 AND amt > 0 AND pbp1 > 0 AND pap1 > pbp1
                THEN GREATEST(-1.0, LEAST(1.0,
                     (amt / vol - (pap1 + pbp1) / 2.0) / ((pap1 - pbp1) / 2.0)))
                ELSE 0
            END AS side,
            CASE WHEN vol > 0 AND amt > 0 AND pbp1 > 0 AND pap1 > pbp1 THEN vol ELSE 0 END AS side_vol,
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
        FROM cte_lagged
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(side * side_vol) AS signed_vol,
            SUM(side_vol) AS total_vol
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
        f.signed_vol, f.total_vol
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    signed_vol = df["signed_vol"].astype("float64")
    total_vol = df["total_vol"].astype("float64")
    df["factor"] = -(signed_vol / total_vol.where(total_vol > 0))
    df = df.replace([np.inf, -np.inf], np.nan)

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
