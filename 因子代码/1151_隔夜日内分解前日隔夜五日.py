def main(datasources, start_date, end_date):
    """【新根因子 方向I：隔夜与日内分解】前一日隔夜收益、日内收益及其分解，日级错位一天后广播到当日各截面。
    【方向I变体 1151：隔夜日内分解前日隔夜五日】

    假设：A 股 T+1 与集合竞价机制下，隔夜收益（前收→次日开盘）与日内收益（开盘→收盘）由不同的交易者群体
    与不同的注意力/流动性条件驱动；两者的差刻画"隔夜跳空被日内修正"的程度，对之后 30 分钟的收益结构有预测力。

    只用 bar1m 的 09:25 集合竞价收盘价（=当日开盘）与 15:00 收盘价。所有统计量在**股票上按交易日**
    计算，再 `shift(1)` 错位一天，截面当天只用到前一交易日及更早的数据，不含任何自身 N 窗口反转阶梯。

    参数：COMPONENT=ovn|intra|diff|ovn5|intra5|diff5。
    """
    import numpy as np
    import pandas as pd
    import dai

    COMPONENT = "ovn5"   # ovn | intra | diff | ovn5 | intra5 | diff5

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    SELECT
        b.instrument_id,
        ai.instrument AS instrument,
        CAST(strftime(b.date, '%Y-%m-%d') || ' 09:25:00' AS TIMESTAMP) AS date,
        strftime(b.date, '%Y-%m-%d') AS trading_day,
        MAX(CASE WHEN strftime(b.date, '%H%M%S') = '092500' THEN CAST(COALESCE(b.close, 0) AS DOUBLE) END) AS d_open,
        MAX(CASE WHEN strftime(b.date, '%H%M%S') = '150000' THEN CAST(COALESCE(b.close, 0) AS DOUBLE) END) AS d_close
    FROM {bar1m} b
    LEFT JOIN all_instruments ai USING (instrument_id)
    WHERE strftime(b.date, '%H%M%S') IN ('092500', '150000')
    GROUP BY 1, 2, 3, 4
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df["date"] = pd.to_datetime(df["date"])
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df = df.dropna(subset=["instrument"])
    df = df.sort_values(["instrument_id", "trading_day"]).reset_index(drop=True)

    grp = df.groupby("instrument_id", sort=False)
    prev_close = grp["d_close"].shift(1)
    df["ovn"] = df["d_open"] / prev_close.where(prev_close > 0) - 1.0
    df["intra"] = df["d_close"] / df["d_open"].where(df["d_open"] > 0) - 1.0
    df["diff"] = df["ovn"] - df["intra"]

    def roll_sum(col, win):
        return (df.groupby("instrument_id", sort=False)[col]
                .rolling(win, min_periods=max(2, win - 2)).sum()
                .reset_index(level=0, drop=True))

    df["ovn5"] = roll_sum("ovn", 5)
    df["intra5"] = roll_sum("intra", 5)
    df["diff5"] = roll_sum("diff", 5)

    stat = df[COMPONENT].astype("float64")
    df["stat"] = stat
    # 错位一天：第 t 个截面的信号只用 t-1 个交易日及更早的统计量。
    df["sig"] = df.groupby("instrument_id", sort=False)["stat"].shift(1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["sig"])
    df = df.drop_duplicates(["instrument", "trading_day"])

    df["factor_raw"] = df.groupby("trading_day")["sig"].rank(pct=True) - 0.5
    sections = ["093000", "100000", "103000", "110000", "113000", "133000", "140000", "143000", "150000"]
    pieces = []
    day_str = df["trading_day"].dt.strftime("%Y-%m-%d")
    for h in sections:
        p = df[["instrument", "factor_raw"]].copy()
        p["date"] = pd.to_datetime(day_str + " " + h[:2] + ":" + h[2:4] + ":00")
        p["factor"] = 0.0 if h == "093000" else p["factor_raw"].values
        pieces.append(p[["date", "instrument", "factor"]])
    out = pd.concat(pieces, ignore_index=True)
    out["factor"] = out["factor"].astype("float64").fillna(0.0)
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
