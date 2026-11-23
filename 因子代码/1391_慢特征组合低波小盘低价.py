def main(datasources, start_date, end_date):
    """【1391：慢特征组合（低波动 + 小盘 + 低价）】单独版，用作合成的慢分量（降换手）。

    思路来自同仓库 sushirui 的盘口锚点系列（146／152）：反转主信号按 1 : 0.5 截面排名融合一个"慢特征组合"后，
    公榜换手从 0.47 降到 0.38–0.43，IC 反而略升，总分 +0.04。本文件只算这个慢分量，交给合成去试权重。

    半小时窗口 (09:30,10:00] … (14:30,15:00]，每只股票按时间连续排列（跨日）：
      低波动：−（最近 40 个窗口 ln(最高 ÷ 最低)² 的均值）（Parkinson 区间方差）
      小盘：−ln(1 + 最近 40 个窗口成交额之和)
      低价：−ln(截面所在窗口的末价)
    三者各自截面排名后等权相加，再截面排名居中。截面 T 只用标签 ≤ T 的窗口（含 T 所在窗口）。
    按自然月分批计算（每月向前多取 15 个自然日供 40 个窗口回看），控制平台整段调用的内存。不读 vwap 表、不读暴露表。
    """
    import gc
    import numpy as np
    import pandas as pd
    import dai

    SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
    bar1m = datasources["bar1m"]

    def _one(s_, e_):
        q0 = (pd.to_datetime(s_) - pd.Timedelta(days=15)).strftime("%Y-%m-%d %H:%M:%S")
        sql = f"""
        WITH b AS (
            SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d,
                CAST(b.close AS DOUBLE) AS c, CAST(b.high AS DOUBLE) AS h, CAST(b.low AS DOUBLE) AS l,
                CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
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
              AND b.close > 0 AND b.high > 0 AND b.low > 0
        )
        SELECT instrument_id, MAX(instrument) AS instrument, d, seg,
            SUM(a) AS amt, MAX(h) AS hi, MIN(l) AS lo, arg_max(c, date) AS c_last
        FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
        """
        w = dai.query(sql, filters={"date": [q0, e_]}, compression=True).df()
        if not len(w):
            return None
        w["instrument"] = w["instrument"].astype(str)
        w["d"] = pd.to_datetime(w["d"])
        w = w.sort_values(["instrument", "d", "seg"]).reset_index(drop=True)
        g = w.groupby("instrument", sort=False)
        with np.errstate(all="ignore"):
            pk = np.log(w["hi"] / w["lo"]) ** 2
            w["c_park"] = -pk.groupby(w["instrument"], sort=False).transform(lambda x: x.rolling(40, min_periods=10).mean())
            w["c_size"] = -np.log1p(g["amt"].transform(lambda x: x.rolling(40, min_periods=10).sum()))
            w["c_lowpx"] = -np.log(w["c_last"].where(w["c_last"] > 0))
        w = w[w["seg"].isin(SLOTS) & (w["d"] >= pd.Timestamp(pd.to_datetime(s_).normalize())) & (w["d"] <= pd.Timestamp(e_))].copy()
        w["date"] = w["d"] + pd.to_timedelta(w["seg"].str[:2].astype(int), "h") + pd.to_timedelta(w["seg"].str[2:].astype(int), "m")
        return w[["date", "instrument", "c_park", "c_size", "c_lowpx"]]

    parts = []
    for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
        _s = max(pd.Timestamp(start_date), _ms.start_time)
        _e = min(pd.Timestamp(end_date), _ms.end_time)
        if _s > _e:
            continue
        p = _one(_s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"))
        if p is not None:
            parts.append(p)
        gc.collect()
    if not parts:
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "instrument"])
    r = df.groupby("date")[["c_park", "c_size", "c_lowpx"]].rank(pct=True)
    df["signal"] = r["c_park"] + r["c_size"] + r["c_lowpx"]
    df["signal"] = df["signal"].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["signal"])
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
