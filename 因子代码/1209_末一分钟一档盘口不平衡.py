def main(datasources, start_date, end_date):
    """【新根因子 第二轮 方向K：截面时刻的微观状态】用**截面时刻之前那一刻**的分钟状态，
    而不是整段累计：最后 1/3/5 分钟收益、最后 1 分钟成交额占当段比例、最后 1 分钟一档盘口不平衡、
    末价相对当段 VWAP 的偏离。

    时间戳约定（本项目已核实）：1 分钟 bar 用**结束时刻**标注（一天最后一条为 150000，
    11:30 一条为 113000），因此"截面 T 的快照"包含标签恰为 T 的那一条 bar，且不含 T 之后的任何 bar。
    全部按股票内当日累计/直取，不跨日、不读 vwap 表、不读暴露表。

    参数：STAT=ret1|ret3|ret5|amtshare1|imb1|devvwap_seg。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "imb1"
    SLOTS = [1000, 1030, 1100, 1130, 1330, 1400, 1430]
    OFF = {"ret1": (0, 1), "ret3": (0, 3), "ret5": (0, 5), "amtshare1": (0, 0),
           "imb1": (0, 0), "devvwap_seg": (0, 0)}[STAT]
    need = {}
    for t in SLOTS:
        hs = {t - k for k in range(0, OFF[1] + 1)}
        need[t] = sorted(hs)
    all_hms = sorted({h for v in need.values() for h in v})
    labels = ["%04d00" % h for h in all_hms]          # HHMM -> HHMMSS 标签（bar 用结束时刻标注）
    seg_case = " OR ".join(["hms = '%s'" % x for x in labels])
    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=5)).strftime("%Y-%m-%d %H:%M:%S")

    sql_min = f"""
    WITH t AS (
        SELECT
            b.instrument_id, ai.instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS amt,
            CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1,
            CAST(COALESCE(b.ask_volume1, 0) AS DOUBLE) AS av1,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close
        FROM {bar1m} b
        LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE (({seg_case}))
    )
    SELECT
        CAST(CONCAT(trading_day, ' ',
             strftime(strptime(LPAD(CAST(hms AS VARCHAR), 6, '0'), '%H%M%S'), '%H:%M:%S')) AS DATETIME) AS date,
        trading_day, instrument_id, MAX(instrument) AS instrument,
        CAST(CAST(hms AS INTEGER) / 100 AS INTEGER) AS hms4,
        MAX(px) AS px, SUM(amt) AS amt, MAX(bv1) AS bv1, MAX(av1) AS av1, MAX(pre_close) AS pre_close
    FROM t GROUP BY trading_day, instrument_id, hms
    """
    mn = dai.query(sql_min, filters={"date": [query_start, end_date]}, compression=True).df()
    if not len(mn):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    mn["instrument"] = mn["instrument"].astype(str)
    mn["instrument_id"] = mn["instrument_id"].astype(str)
    mn["hms"] = mn["hms4"].astype(int)
    pre = mn.groupby(["trading_day", "instrument_id"])["pre_close"].max()

    sql_seg = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS amt,
            CAST(COALESCE(b.volume, 0) AS DOUBLE) AS vol,
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
        WHERE strftime(b.date, '%H%M%S') > '090000'
    )
    SELECT
        CAST(CONCAT(trading_day, ' ',
             strftime(strptime(LPAD(CAST(time_segment AS VARCHAR), 6, '0'), '%H%M%S'), '%H:%M:%S')) AS DATETIME) AS date,
        trading_day, instrument_id, time_segment, SUM(amt) AS seg_amt, SUM(vol) AS seg_vol
    FROM cte_raw WHERE time_segment != -1 GROUP BY trading_day, instrument_id, time_segment
    """
    sg = dai.query(sql_seg, filters={"date": [query_start, end_date]}, compression=True).df()
    sg["instrument_id"] = sg["instrument_id"].astype(str)
    sg["hms"] = (sg["time_segment"].astype(int) // 100).astype(int)
    sg["seg_vwap"] = sg["seg_amt"] / sg["seg_vol"].where(sg["seg_vol"] > 0)

    rows = []
    for t in SLOTS:
        sub = mn[mn["hms"].isin(need[t])].copy()
        piv = sub.pivot_table(index=["trading_day", "instrument_id"], columns="hms",
                              values=["px", "amt", "bv1", "av1"], aggfunc="max")
        piv.columns = ["%s_%d" % (a, b) for a, b in piv.columns]
        # 某些分钟（尤其冷门股/停牌前后）可能整列缺失，先补齐成 NaN 列，避免 .get 返回 None
        piv = piv.reindex(columns=["%s_%d" % (a, h) for a in ("px", "amt", "bv1", "av1") for h in need[t]])
        piv = piv.reset_index()
        if STAT in ("ret1", "ret3", "ret5"):
            k = OFF[1]
            cp = piv.get("px_%d" % t)
            c0 = piv.get("px_%d" % (t - k))
            sig = (cp - c0) / pre.reindex(pd.MultiIndex.from_frame(piv[["trading_day", "instrument_id"]])).to_numpy()
        elif STAT == "amtshare1":
            a1 = piv.get("amt_%d" % t)
            seg = sg[sg["hms"] == t].set_index(["trading_day", "instrument_id"])["seg_amt"]
            sig = a1.to_numpy() / seg.reindex(pd.MultiIndex.from_frame(piv[["trading_day", "instrument_id"]])).to_numpy()
        elif STAT == "imb1":
            b1 = piv.get("bv1_%d" % t).to_numpy(float)
            a1 = piv.get("av1_%d" % t).to_numpy(float)
            sig = (b1 - a1) / np.where((b1 + a1) > 0, b1 + a1, np.nan)
        else:
            cp = piv.get("px_%d" % t)
            vw = sg[sg["hms"] == t].set_index(["trading_day", "instrument_id"])["seg_vwap"]
            p0 = pre.reindex(pd.MultiIndex.from_frame(piv[["trading_day", "instrument_id"]])).to_numpy()
            sig = (cp.to_numpy(float) - vw.reindex(pd.MultiIndex.from_frame(piv[["trading_day", "instrument_id"]])).to_numpy()) / p0
        inst = piv[["trading_day", "instrument_id"]].copy()
        inst["date"] = pd.to_datetime(inst["trading_day"]) + pd.to_timedelta(t // 100, "h") + pd.to_timedelta(t % 100, "m")
        inst["signal"] = pd.Series(np.asarray(sig, dtype=float))
        rows.append(inst[["date", "trading_day", "instrument_id", "signal"]])
    df = pd.concat(rows, ignore_index=True)
    df = df.merge(mn[["trading_day", "instrument_id", "instrument"]].drop_duplicates(),
                  on=["trading_day", "instrument_id"], how="left")
    df["signal"] = df["signal"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["instrument", "signal"]).drop_duplicates(["date", "instrument"])
    df["factor"] = df.groupby("date")["signal"].rank(pct=True) - 0.5
    df["factor"] = df["factor"].fillna(0.0)
    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)].drop_duplicates(["date", "instrument"])
    out = df[["date", "instrument", "factor"]].copy()
    out["date"] = pd.to_datetime(out["date"])
    out["instrument"] = out["instrument"].astype(str)
    out = _pool.merge(out.drop_duplicates(["date", "instrument"]), on=["date", "instrument"], how="left")
    out["factor"] = out["factor"].astype("float64").fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
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
