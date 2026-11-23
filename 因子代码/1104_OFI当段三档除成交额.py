def main(datasources, start_date, end_date):
    """【新根因子 方向A：订单流不平衡 OFI】挂单量的变化（Cont–Kukanov–Stoikov 口径）当日累计，除以同日累计成交额。
    【方向A变体 1104：OFI当段三档除成交额】

    假设：买卖盘挂单的**变化**（不是挂单水平）反映知情交易者的主动进出——买盘加价加量、卖盘撤单都是买压；
    这部分压力会延续到之后的截面。只用挂单价格与数量的变化，不含任何自身收益反转或 K 线形态分量。

    参数：OFI_DEPTH=1|3（1 档 / 1–3 档加权）、OFI_WINDOW=day|seg（当日累计到该截面 / 只取本时间段）、
    OFI_DENOM=amt|mcap（同日累计成交额 / 前一交易日流通市值）、OFI_LOWPASS=0|alpha。

    实现：OFI 的逐分钟差分与分量和全部在 SQL 里按分钟窗口完成（一条 SQL 同时给出 1 档与 2–3 档分量），
    pandas 只处理按时间段聚合后的几十万行；六个变体共用同一条 SQL，平台缓存可复用。
    """
    import numpy as np
    import pandas as pd
    import dai

    OFI_DEPTH = 3          # 1 或 3
    OFI_WINDOW = "seg"     # day 或 seg
    OFI_DENOM = "amt"      # amt 或 mcap（mcap 用暴露表前一交易日值，显式 shift(1)）
    OFI_LOWPASS = 0.0      # 0 表示不低通

    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument AS instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(b.bid_price2, 0) AS DOUBLE) AS bp2,
            CAST(COALESCE(b.bid_price3, 0) AS DOUBLE) AS bp3,
            CAST(COALESCE(b.ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(b.ask_price2, 0) AS DOUBLE) AS ap2,
            CAST(COALESCE(b.ask_price3, 0) AS DOUBLE) AS ap3,
            CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1,
            CAST(COALESCE(b.bid_volume2, 0) AS DOUBLE) AS bv2,
            CAST(COALESCE(b.bid_volume3, 0) AS DOUBLE) AS bv3,
            CAST(COALESCE(b.ask_volume1, 0) AS DOUBLE) AS av1,
            CAST(COALESCE(b.ask_volume2, 0) AS DOUBLE) AS av2,
            CAST(COALESCE(b.ask_volume3, 0) AS DOUBLE) AS av3,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS amt
        FROM {bar1m} b
        LEFT JOIN all_instruments ai USING (instrument_id)
    ),
    cte_lag AS (
        SELECT
            instrument_id, instrument, trading_day, hms, amt,
            bp1, bp2, bp3, ap1, ap2, ap3, bv1, bv2, bv3, av1, av2, av3,
            LAG(bp1) OVER w AS bp1_p, LAG(bv1) OVER w AS bv1_p,
            LAG(bp2) OVER w AS bp2_p, LAG(bv2) OVER w AS bv2_p,
            LAG(bp3) OVER w AS bp3_p, LAG(bv3) OVER w AS bv3_p,
            LAG(ap1) OVER w AS ap1_p, LAG(av1) OVER w AS av1_p,
            LAG(ap2) OVER w AS ap2_p, LAG(av2) OVER w AS av2_p,
            LAG(ap3) OVER w AS ap3_p, LAG(av3) OVER w AS av3_p
        FROM cte_raw
        WINDOW w AS (PARTITION BY instrument_id ORDER BY trading_day, hms)
    ),
    cte_ofi AS (
        SELECT
            instrument_id, instrument, trading_day, hms, amt,
            (CASE WHEN bp1 > bp1_p THEN bv1 WHEN bp1 < bp1_p THEN -bv1_p ELSE 0 END
             + CASE WHEN ap1 < ap1_p THEN -av1 WHEN ap1 > ap1_p THEN av1_p ELSE 0 END) AS ofi_d1,
            (0.5 * (CASE WHEN bp2 > bp2_p THEN bv2 WHEN bp2 < bp2_p THEN -bv2_p ELSE 0 END
                    + CASE WHEN ap2 < ap2_p THEN -av2 WHEN ap2 > ap2_p THEN av2_p ELSE 0 END)
             + (1.0 / 3.0) * (CASE WHEN bp3 > bp3_p THEN bv3 WHEN bp3 < bp3_p THEN -bv3_p ELSE 0 END
                              + CASE WHEN ap3 < ap3_p THEN -av3 WHEN ap3 > ap3_p THEN av3_p ELSE 0 END)) AS ofi_d23
        FROM cte_lag
    ),
    cte_seg AS (
        SELECT
            instrument_id, instrument, trading_day, hms, amt, ofi_d1, ofi_d23,
            CASE
                WHEN hms <= '093000' THEN 93000
                WHEN hms <= '100000' THEN 100000
                WHEN hms <= '103000' THEN 103000
                WHEN hms <= '110000' THEN 110000
                WHEN hms <= '113000' THEN 113000
                WHEN hms <= '133000' THEN 133000
                WHEN hms <= '140000' THEN 140000
                WHEN hms <= '143000' THEN 143000
                ELSE 150000
            END AS seg
        FROM cte_ofi
    )
    SELECT
        s.instrument_id AS instrument_id,
        s.instrument AS instrument,
        s.trading_day AS trading_day,
        s.seg AS seg,
        MAX(CAST(s.trading_day || ' 09:25:00' AS TIMESTAMP)) AS date,
        SUM(s.ofi_d1) AS ofi_d1,
        SUM(s.ofi_d23) AS ofi_d23,
        SUM(s.amt) AS amt_seg
    FROM cte_seg s
    WHERE s.seg <> 150000
    GROUP BY s.instrument_id, s.instrument, s.trading_day, s.seg
    """
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    df["instrument"] = df["instrument"].astype(str)
    df = df.sort_values(["instrument_id", "trading_day", "seg"]).reset_index(drop=True)

    if OFI_DEPTH == 1:
        df["ofi"] = df["ofi_d1"]
    else:
        df["ofi"] = df["ofi_d1"] + df["ofi_d23"]

    if OFI_WINDOW == "day":
        df["num"] = df.groupby(["instrument_id", "trading_day"], sort=False)["ofi"].cumsum()
        df["den_sum"] = df.groupby(["instrument_id", "trading_day"], sort=False)["amt_seg"].cumsum()
    else:
        df["num"] = df["ofi"]
        df["den_sum"] = df["amt_seg"]

    if OFI_DENOM == "mcap":
        # 暴露表错位：只用严格早于截面日期的最近一条（前一交易日及更早）。
        exp = dai.query("SELECT date, instrument, float_market_cap FROM cpt_jyc_2026_exposure",
                        filters={"date": [query_start, end_date]}, compression=True).df()
        exp["date"] = pd.to_datetime(exp["date"])
        exp = exp.sort_values(["instrument", "date"]).drop_duplicates(["instrument", "date"])
        exp["mcap"] = exp.groupby("instrument", sort=False)["float_market_cap"].shift(1)
        merged = df.merge(exp[["instrument", "date", "mcap"]], left_on=["instrument", "trading_day"],
                          right_on=["instrument", "date"], how="left")
        df = merged
        den = merged["mcap"]
    else:
        den = df["den_sum"]

    df["signal"] = (df["num"] / den.where(den > 0)).replace([np.inf, -np.inf], np.nan)

    seg_map = {93000: "093000", 100000: "100000", 103000: "103000", 110000: "110000",
               113000: "113000", 133000: "133000", 140000: "140000", 143000: "143000"}
    df["hm"] = df["seg"].map(seg_map)
    df = df.dropna(subset=["instrument", "signal", "hm"])
    df["date"] = pd.to_datetime(df["trading_day"].dt.strftime("%Y-%m-%d") + " " +
                                df["hm"].str[:2] + ":" + df["hm"].str[2:4] + ":00")
    out = df.drop_duplicates(["date", "instrument"]).copy()

    if OFI_LOWPASS:
        out = out.sort_values(["instrument", "date"])
        out["signal"] = out.groupby("instrument", sort=False)["signal"].transform(
            lambda x: x.ewm(alpha=float(OFI_LOWPASS), adjust=False).mean())

    out["factor"] = out.groupby("date")["signal"].rank(pct=True) - 0.5
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
