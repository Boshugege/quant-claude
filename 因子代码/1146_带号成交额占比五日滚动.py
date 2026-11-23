def main(datasources, start_date, end_date):
    """【新根因子 方向H：主动买卖】tick rule 带号成交额占比与"大单分钟"带号净额占比。
    【方向H变体 1146：带号成交额占比五日滚动】

    假设：主动买入的持续性（带号成交额失衡）与"大单（单笔额大的分钟）方向"反映知情交易的持续进入，
    对之后 30 分钟收益有预测力。带号用 tick rule（本分钟收益符号），大单用"分钟成交额 ≥ 100 万元"近似
    （只有分钟 bar，没有逐笔，属于必要的近似，已记入局限）。

    只用分钟 close/amount。占比在 SQL 里按段算出（分子分母都是可加和），pandas 里累加。
    不含反转阶梯、K 线形态、泄漏分量。

    参数：STAT=net|big_net、WINDOW=seg|day、ROLL_DAYS=0|5。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "net"
    WINDOW = "seg"
    ROLL_DAYS = 5

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument AS instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS amt,
            CASE
                WHEN strftime(b.date, '%H%M%S') > '090000' AND strftime(b.date, '%H%M%S') <= '093000' THEN 93000
                WHEN strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '100000' THEN 100000
                WHEN strftime(b.date, '%H%M%S') > '100000' AND strftime(b.date, '%H%M%S') <= '103000' THEN 103000
                WHEN strftime(b.date, '%H%M%S') > '103000' AND strftime(b.date, '%H%M%S') <= '110000' THEN 110000
                WHEN strftime(b.date, '%H%M%S') > '110000' AND strftime(b.date, '%H%M%S') <= '113000' THEN 113000
                WHEN strftime(b.date, '%H%M%S') > '130000' AND strftime(b.date, '%H%M%S') <= '133000' THEN 133000
                WHEN strftime(b.date, '%H%M%S') > '133000' AND strftime(b.date, '%H%M%S') <= '140000' THEN 140000
                WHEN strftime(b.date, '%H%M%S') > '140000' AND strftime(b.date, '%H%M%S') <= '143000' THEN 143000
                WHEN strftime(b.date, '%H%M%S') > '143000' AND strftime(b.date, '%H%M%S') <= '150000' THEN 150000
                ELSE -1
            END AS time_segment
        FROM {bar1m} b
        LEFT JOIN all_instruments ai USING (instrument_id)
    ),
    cte_min AS (
        SELECT
            instrument_id, instrument, trading_day, hms, time_segment, amt,
            CASE WHEN px_close > 0 AND pre_close > 0 THEN
                LOG(px_close / COALESCE(NULLIF(LAG(px_close) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms), 0), pre_close))
            ELSE NULL END AS r
        FROM cte_raw
        WHERE time_segment != -1
    )
    SELECT
        CAST(CONCAT(
            trading_day, ' ',
            strftime(strptime(LPAD(CAST(time_segment AS VARCHAR), 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        trading_day,
        time_segment,
        instrument_id,
        MAX(instrument) AS instrument,
        COUNT(r) AS n,
        SUM(CASE WHEN r > 0 THEN amt ELSE 0 END) AS pos_amt,
        SUM(CASE WHEN r < 0 THEN amt ELSE 0 END) AS neg_amt,
        SUM(CASE WHEN r > 0 AND amt >= 1000000 THEN amt ELSE 0 END) AS big_pos_amt,
        SUM(CASE WHEN r < 0 AND amt >= 1000000 THEN amt ELSE 0 END) AS big_neg_amt
    FROM cte_min
    GROUP BY trading_day, time_segment, instrument_id
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    if not len(df):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    df["instrument"] = df["instrument"].astype(str)
    df["instrument_id"] = df["instrument_id"].astype(str)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument_id", "trading_day", "time_segment"]).reset_index(drop=True)

    keys = ["instrument_id", "trading_day"]
    cols = ["pos_amt", "neg_amt", "big_pos_amt", "big_neg_amt"]
    if WINDOW == "day":
        for c in cols:
            df["c_" + c] = df.groupby(keys, sort=False)[c].cumsum()
    else:
        for c in cols:
            df["c_" + c] = df[c]
    if STAT == "net":
        num = df["c_pos_amt"] - df["c_neg_amt"]
        den = df["c_pos_amt"] + df["c_neg_amt"]
    else:
        num = df["c_big_pos_amt"] - df["c_big_neg_amt"]
        den = df["c_big_pos_amt"] + df["c_big_neg_amt"]
    df["stat"] = num / den.where(den > 0)
    df = df.dropna(subset=["instrument", "instrument_id"])

    if ROLL_DAYS:
        df["signal"] = df.groupby(["instrument_id", "time_segment"], sort=False)["stat"].transform(
            lambda x: x.rolling(ROLL_DAYS, min_periods=max(2, ROLL_DAYS - 1)).mean())
    else:
        df["signal"] = df["stat"]

    out = df.dropna(subset=["signal"])[["date", "instrument", "signal"]]
    out = out.drop_duplicates(["date", "instrument"]).copy()
    out["factor"] = out.groupby("date")["signal"].rank(pct=True) - 0.5
    out["factor"] = out["factor"].fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    # 平台骨架：只保留 8 个约定截面（0930/1000/1030/1100/1130/1330/1400/1430），
    # 并以股票池 (date, instrument) 为全量骨架（信号缺失处填 0），与 485 等已交付因子同口径。
    # 这同时避免了按季度调用时相邻批次重叠行造成的重复键与越界行。
    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)]
    _pool = _pool.drop_duplicates(["date", "instrument"])
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype(str)
    out = out.drop_duplicates(["date", "instrument"])
    out = _pool.merge(out[["date", "instrument", "factor"]], on=["date", "instrument"], how="left")
    out["factor"] = out["factor"].astype("float64").fillna(0.0)
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


if __name__ == "__main__":
    import pandas as pd
    from bigmodule import M
    import structlog

    logger = structlog.get_logger()
    datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
    start_date = "2020-01-01 00:00:00"
    end_date = "2020-03-01 23:59:59"
    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
