def main(datasources, start_date, end_date):
    """【新根因子 方向J：竞价信息】09:25 集合竞价的委托/成交信息构造的截面信号，当日广播到各截面。
    【方向J变体 1156：竞价跳空乘异常额】

    假设：开盘前集合竞价的成交量、跳空幅度与相对自身历史的异常程度，反映开盘前知情交易者与散户的
    信息与情绪；竞价放量或高开在 T+1 与涨跌停约束下未必马上被完全定价，对之后 30 分钟的收益有预测力。

    只用 bar1m 的 09:25 竞价 bar（amount/volume/close/pre_close）与前一交易日的全天成交额；
    相对基准一律"股票内 shift(1) 后再 rolling"，只用严格早于当天的日子。当日信号在 09:25 已知。

    参数：COMPONENT=amt_ratio|gap|gap_x_amt|amt_share|vol_ratio|gap_x_share。
    """
    import numpy as np
    import pandas as pd
    import dai

    COMPONENT = "gap_x_amt"   # amt_ratio | gap | gap_x_amt | amt_share | vol_ratio | gap_x_share

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_auc AS (
        SELECT
            b.instrument_id AS instrument_id,
            ai.instrument AS instrument,
            b.date AS date,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS auc_px,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS auc_amt,
            CAST(COALESCE(b.volume, 0) AS DOUBLE) AS auc_vol,
            CAST(COALESCE(b.deal_number, 0) AS DOUBLE) AS auc_deals
        FROM {bar1m} b
        LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE strftime(b.date, '%H%M%S') = '092500'
    ),
    cte_day AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            SUM(CAST(COALESCE(amount, 0) AS DOUBLE)) AS day_amt
        FROM {bar1m}
        WHERE strftime(date, '%H%M%S') BETWEEN '093100' AND '150000'
        GROUP BY 1, 2
    )
    SELECT a.*, d.day_amt
    FROM cte_auc a
    LEFT JOIN cte_day d ON a.instrument_id = d.instrument_id AND a.trading_day = d.trading_day
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df["date"] = pd.to_datetime(df["date"])
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df = df.dropna(subset=["instrument"]).sort_values(["instrument_id", "trading_day"]).reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)

    def base(col, win=5):
        prev = df.groupby("instrument_id", sort=False)[col].shift(1)
        base = prev.groupby(df["instrument_id"], sort=False).rolling(win, min_periods=2).mean() \
            .reset_index(level=0, drop=True)
        return base

    amt_base = base("auc_amt")
    vol_base = base("auc_vol")
    prev_day_amt = df.groupby("instrument_id", sort=False)["day_amt"].shift(1)

    df["gap"] = df["auc_px"] / df["pre_close"].where(df["pre_close"] > 0) - 1.0
    df["amt_ratio"] = df["auc_amt"] / amt_base.where(amt_base > 0)
    df["vol_ratio"] = df["auc_vol"] / vol_base.where(vol_base > 0)
    df["amt_share"] = df["auc_amt"] / prev_day_amt.where(prev_day_amt > 0)
    df["gap_x_amt"] = df["gap"] * np.log1p(df["amt_ratio"].clip(lower=0))
    df["gap_x_share"] = df["gap"] * np.log1p(df["amt_share"].clip(lower=0))

    df["sig"] = df[COMPONENT].astype("float64").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["sig"]).drop_duplicates(["instrument", "trading_day"])
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
