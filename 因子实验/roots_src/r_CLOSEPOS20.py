def main(datasources, start_date, end_date):
    """【收盘位置 CLOSEPOS20】(当日收盘价 ÷ 当日 8 个半小时段收盘价的最高者 − 1) 在过去 20 个交易日的均值（至少 10 日，按同一股票自己的交易日滚动，错开 1 日）。
    同一交易日内各截面（10:00…14:30）取值相同（"日内不变锚"，压换手），逐截面百分位排名 − 0.5，09:30 置 0。口径同 因子实验/k1_anchor.py。
    30 分钟段与 1232 同一段 SQL：1 分钟 bar 聚成 (09:30,10:00] … (14:30,15:00]。按自然月分批调用，控制内存。不用成交笔数。
    """
    import pandas as pd
    parts = []
    for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
        _s = max(pd.Timestamp(start_date), _ms.start_time)
        _e = min(pd.Timestamp(end_date), _ms.end_time)
        if _s > _e:
            continue
        parts.append(_cp_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"), 20))
    out = pd.concat(parts, ignore_index=True)
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def _cp_impl(datasources, start_date, end_date, N):
    import numpy as np
    import pandas as pd
    import dai

    SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
    SEGS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
    bar1m = datasources["bar1m"]
    q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    sql_seg = f"""
    WITH b AS (
        SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d,
            CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c,
            CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
            CASE
                WHEN strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '100000' THEN '1000'
                WHEN strftime(b.date, '%H%M%S') > '100000' AND strftime(b.date, '%H%M%S') <= '103000' THEN '1030'
                WHEN strftime(b.date, '%H%M%S') > '103000' AND strftime(b.date, '%H%M%S') <= '110000' THEN '1100'
                WHEN strftime(b.date, '%H%M%S') > '110000' AND strftime(b.date, '%H%M%S') <= '113000' THEN '1130'
                WHEN strftime(b.date, '%H%M%S') > '130000' AND strftime(b.date, '%H%M%S') <= '133000' THEN '1330'
                WHEN strftime(b.date, '%H%M%S') > '133000' AND strftime(b.date, '%H%M%S') <= '140000' THEN '1400'
                WHEN strftime(b.date, '%H%M%S') > '140000' AND strftime(b.date, '%H%M%S') <= '143000' THEN '1430'
                WHEN strftime(b.date, '%H%M%S') > '143000' AND strftime(b.date, '%H%M%S') <= '150000' THEN '1500'
            END AS seg
        FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '150000'
          AND b.close > 0 AND b.open > 0
    )
    SELECT MAX(date) AS date, instrument_id, MAX(instrument) AS instrument, d, seg,
        SUM(a) AS amt, SUM(v) AS vol, arg_max(c, date) AS c_last, arg_min(o, date) AS o_first
    FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
    """
    sg = dai.query(sql_seg, filters={"date": [q0, end_date]}, compression=True).df()
    if not len(sg):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    sg["instrument"] = sg["instrument"].astype(str)
    sg["d"] = pd.to_datetime(sg["d"])
    cl = sg.pivot_table(index=["instrument", "d"], columns="seg", values="c_last", aggfunc="last").reindex(columns=SEGS).sort_index()
    with np.errstate(all="ignore"):
        cp = cl["1500"] / cl[SEGS].max(axis=1) - 1.0
    g = cp.groupby(level=0, sort=False)
    m = g.rolling(N, min_periods=N // 2).mean()
    m.index = m.index.droplevel(0)
    m = m.reindex(cp.index).groupby(level=0, sort=False).shift(1)            # 只用之前的交易日
    d0 = pd.Timestamp(pd.to_datetime(start_date).normalize())
    s0 = m[m.index.get_level_values(1) >= d0].rename("signal").reset_index()
    rows = []
    for t in SLOTS:
        s = s0.copy()
        s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
        rows.append(s[["date", "instrument", "signal"]])
    df = pd.concat(rows, ignore_index=True)
    df["signal"] = df["signal"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["instrument", "signal"]).drop_duplicates(["date", "instrument"])
    df["factor"] = df.groupby("date")["signal"].rank(pct=True) - 0.5
    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)].drop_duplicates(["date", "instrument"])
    out = _pool.merge(df[["date", "instrument", "factor"]], on=["date", "instrument"], how="left")
    out["factor"] = out["factor"].astype("float64").fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)
