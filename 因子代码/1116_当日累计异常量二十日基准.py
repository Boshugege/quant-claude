def main(datasources, start_date, end_date):
    """【新根因子 方向C：同时段异常量】本段成交额 ÷ 过去 N 日同一时段成交额均值（只用过去日），可乘本段收益符号。
    【方向C变体 1116：当日累计异常量二十日基准】

    假设：同一时段相对自己历史的异常放量是信息事件（知情交易或被动成交的集中释放），
    它本身反映关注度冲击，与**本段收益方向**结合后能区分"放量上涨"与"放量下跌"，对之后 30 分钟的收益有预测力。

    只用到分钟行情的成交额、成交笔数、末价与前收，以及暴露表（本族默认不读）。不含反转阶梯、K 线形态、泄漏分量。
    分段聚合在 SQL 里做（内存友好）。

    参数：BASE_DAYS=基准天数（只用严格早于当天的日子）、WINDOW=seg|day（当段 / 当日累计到该段）、
    SIGNED=0|1|2（0=原始异常量、1=乘本段收益符号、2=乘本段收益）、METRIC=amt|deals。
    """
    import numpy as np
    import pandas as pd
    import dai

    BASE_DAYS = 20          # 基准窗口：过去多少个交易日
    WINDOW = "day"          # seg | day（day = 当日累计到该段）
    SIGNED = 0              # 0=原始；1=乘符号；2=乘收益
    METRIC = "amt"          # amt | deals

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    amt_col = "CAST(COALESCE(amount, 0) AS DOUBLE)" if METRIC == "amt" else "CAST(COALESCE(deal_number, 0) AS DOUBLE)"
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close,
            {amt_col} AS qty
        FROM {bar1m} b
        LEFT JOIN all_instruments ai USING (instrument_id)
    ),
    cte_seg AS (
        SELECT
            instrument_id, instrument, trading_day, hms, px_close, pre_close, qty,
            CASE WHEN px_close > 0 THEN px_close ELSE 0 END AS px,
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
    )
    SELECT
        CAST(CONCAT(
            s.trading_day, ' ',
            strftime(strptime(LPAD(CAST(s.time_segment AS VARCHAR), 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        s.trading_day,
        s.time_segment,
        s.instrument_id,
        MAX(s.instrument) AS instrument,
        SUM(s.qty) AS seg_qty,
        MAX(CASE WHEN CAST(s.hms AS INTEGER) = s.time_segment
                  OR (s.time_segment = 93000 AND s.hms = '092500') THEN s.px ELSE 0 END) AS last_px,
        MAX(s.pre_close) AS pre_close
    FROM cte_seg s
    WHERE s.time_segment != -1
    GROUP BY s.trading_day, s.time_segment, s.instrument_id
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    if not len(df):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    df["instrument"] = df["instrument"].astype(str)
    df["instrument_id"] = df["instrument_id"].astype(str)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["instrument_id", "trading_day", "time_segment"]).reset_index(drop=True)

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    gs = df.groupby(["instrument_id", "trading_day"], sort=False)
    px_series = df["last_px"].where(df["last_px"] > 0)
    day_pre = gs["pre_close"].transform("max")
    px_lag1 = px_series.groupby([df["instrument_id"], df["trading_day"]], sort=False).shift(1)
    px_lag1 = px_lag1.where(px_lag1 > 0, day_pre)
    df["seg_ret"] = safe_log_ratio(px_series, px_lag1)

    if WINDOW == "day":
        df["qty_use"] = df.groupby(["instrument_id", "trading_day"], sort=False)["seg_qty"].cumsum()
    else:
        df["qty_use"] = df["seg_qty"]

    # 基准：同一 (股票, 时段) 序列上，只要**严格早于当天**的过去 BASE_DAYS 个交易日
    gg = df.groupby(["instrument_id", "time_segment"], sort=False)
    past = gg["qty_use"].shift(1)
    base = past.groupby([df["instrument_id"], df["time_segment"]], sort=False).rolling(
        BASE_DAYS, min_periods=max(3, BASE_DAYS // 4)).mean().reset_index(level=[0, 1], drop=True)
    df["base_qty"] = base.reindex(df.index)
    df["abn"] = df["qty_use"] / df["base_qty"].where(df["base_qty"] > 0)

    if SIGNED == 1:
        df["signal"] = df["abn"] * np.sign(df["seg_ret"])
    elif SIGNED == 2:
        df["signal"] = df["abn"] * df["seg_ret"]
    else:
        df["signal"] = df["abn"]
    df["signal"] = df["signal"].replace([np.inf, -np.inf], np.nan)

    out = df.dropna(subset=["instrument", "signal"])[["date", "instrument", "signal"]]
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
