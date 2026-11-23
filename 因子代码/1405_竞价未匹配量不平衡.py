def main(datasources, start_date, end_date):
    """【1405：竞价未匹配量不平衡】新根单独版（公榜探针），STAT = O_imb，方向 正向。
    根 = 1400_集合竞价框架.py 原样复用；按自然月分批调用，控制平台整段调用时的内存。
    """
    import pandas as pd
    parts = []
    for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
        _s = max(pd.Timestamp(start_date), _ms.start_time)
        _e = min(pd.Timestamp(end_date), _ms.end_time)
        if _s > _e:
            continue
        parts.append(_root_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"), "O_imb"))
    out = pd.concat(parts, ignore_index=True)
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def _root_impl(datasources, start_date, end_date, STAT):
    """【新根因子 第十轮：集合竞价与隔夜框架（1400 起）】一份代码覆盖开盘集合竞价、隔夜、前一日收盘集合竞价、首半小时，STAT 选一个统计量。
    同一段 SQL 被所有变体共用（评测机按 SQL 文本缓存），变体只改 STAT 一行。

    数据口径（1 分钟表）：09:16–09:24 的 bar 成交量为 0，收盘价是集合竞价的虚拟撮合价；一档买、卖价量相同（= 撮合价与可撮合量），
    未匹配量只出现在二档的一侧（价格记 0）；09:25 的 bar 是开盘集合竞价的成交（约 55% 的股票这一 bar 有成交量，其余为 0，
    竞价量类变体只覆盖前者）；15:00 的 bar 含收盘集合竞价成交，14:57 的 bar 收盘价是收盘竞价前的最后连续竞价价。
    当日只用 10:00 及之前的 bar（所有截面都在 10:00 及之后）；当日收盘、全天成交量等一律取前一交易日（按股票 shift(1)）。
    每个交易日一个值，7 个截面相同（日内常数，换手低）。不读 vwap 表、不读暴露表。

    STAT：
      O_gap     隔夜跳空：开盘竞价价 ÷ 前一日收盘 − 1
      O_aucv20  开盘竞价成交量 ÷ 过去 20 个交易日开盘竞价成交量均值（对数）
      O_aucs    开盘竞价成交量 ÷ 前一日全天成交量
      O_pre     竞价后段价格变动：开盘竞价价 ÷ 09:20 虚拟撮合价 − 1（09:20 之后不能撤单）
      O_imb     09:24 未匹配量不平衡：(未匹配买 − 未匹配卖) ÷ (可撮合量 + 未匹配买 + 未匹配卖)
      O_cauc    前一日收盘竞价的价格冲击：前一日收盘 ÷ 前一日 14:57 价 − 1
      O_caucs   前一日收盘竞价成交量占全天比例
      O_f30     首半小时收益：10:00 价 ÷ 开盘竞价价 − 1
      O_pre1    竞价前段（可撤单阶段）价格变动：09:20 虚拟撮合价 ÷ 09:16 虚拟撮合价 − 1
      O_lim1    前一日收盘涨停（1）／跌停（−1）／其他（0）；涨跌幅限制按代码：688、300、301 开头 20%，8、4 开头 30%，其余 10%（ST 无法识别，按 10%）
      O_lim5    过去 5 个交易日收盘涨停次数 − 跌停次数
    """
    import numpy as np
    import pandas as pd
    import dai

    SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
    bar1m = datasources["bar1m"]
    q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=45)).strftime("%Y-%m-%d %H:%M:%S")   # 20 个交易日回看 + 前一日
    sql = f"""
    WITH b AS (
        SELECT b.instrument_id, ai.instrument, CAST(b.date AS DATE) AS d, strftime(b.date, '%H%M%S') AS t,
            CAST(b.close AS DOUBLE) AS c, CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v,
            CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1, CAST(COALESCE(b.bid_volume2, 0) AS DOUBLE) AS bv2,
            CAST(COALESCE(b.ask_volume2, 0) AS DOUBLE) AS av2
        FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE b.close > 0
    )
    SELECT instrument_id, MAX(instrument) AS instrument, d,
        MAX(CASE WHEN t = '091600' THEN c END) AS px0916, MAX(CASE WHEN t = '092000' THEN c END) AS px0920,
        MAX(CASE WHEN t = '092400' THEN bv1 END) AS m0924, MAX(CASE WHEN t = '092400' THEN bv2 END) AS ub0924,
        MAX(CASE WHEN t = '092400' THEN av2 END) AS ua0924,
        MAX(CASE WHEN t = '092500' THEN c END) AS apx, MAX(CASE WHEN t = '092500' THEN v END) AS aucv,
        MAX(CASE WHEN t = '100000' THEN c END) AS c1000,
        MAX(CASE WHEN t = '145700' THEN c END) AS c1457,
        MAX(CASE WHEN t = '150000' THEN c END) AS cls, MAX(CASE WHEN t = '150000' THEN v END) AS cauv,
        SUM(v) AS dv
    FROM b GROUP BY instrument_id, d
    """
    g = dai.query(sql, filters={"date": [q0, end_date]}, compression=True).df()
    if not len(g):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    g["instrument"] = g["instrument"].astype(str)
    g["d"] = pd.to_datetime(g["d"])
    g = g.sort_values(["instrument", "d"]).reset_index(drop=True)
    grp = g.groupby("instrument", sort=False)
    with np.errstate(all="ignore"):
        prev_cls = grp["cls"].shift(1)                                   # 只用前一交易日的收盘、全天量、收盘竞价
        prev_dv = grp["dv"].shift(1)
        prev_c1457 = grp["c1457"].shift(1)
        prev_cauv = grp["cauv"].shift(1)
        mu20 = grp["aucv"].transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
        tot = g["m0924"] + g["ub0924"] + g["ua0924"]
        _code = g["instrument"].str[:3]
        _lim = np.where(_code.isin(["688", "300", "301"]), 0.20, np.where(g["instrument"].str[:1].isin(["8", "4"]), 0.30, 0.10))
        _pc = grp["cls"].shift(1)                                        # 涨跌停判定用"当日收盘 vs 前一日收盘"，再整体后移一天才用
        _up = (g["cls"] >= np.round(_pc * (1 + _lim), 2) - 1e-6).astype(float).where(_pc > 0)
        _dn = (g["cls"] <= np.round(_pc * (1 - _lim), 2) + 1e-6).astype(float).where(_pc > 0)
        g["_lstate"] = _up - _dn
        lim1 = g.groupby("instrument", sort=False)["_lstate"].shift(1)
        lim5 = g.groupby("instrument", sort=False)["_lstate"].transform(lambda s: s.shift(1).rolling(5, min_periods=3).sum())
        sig = {
            "O_gap": g["apx"] / prev_cls.where(prev_cls > 0) - 1,
            "O_aucv20": np.log(g["aucv"].where(g["aucv"] > 0) / mu20.where(mu20 > 0)),
            "O_aucs": g["aucv"] / prev_dv.where(prev_dv > 0),
            "O_pre": g["apx"] / g["px0920"].where(g["px0920"] > 0) - 1,
            "O_imb": (g["ub0924"] - g["ua0924"]) / tot.where(tot > 0),
            "O_cauc": prev_cls / prev_c1457.where(prev_c1457 > 0) - 1,
            "O_caucs": prev_cauv / prev_dv.where(prev_dv > 0),
            "O_f30": g["c1000"] / g["apx"].where(g["apx"] > 0) - 1,
            "O_pre1": g["px0920"] / g["px0916"].where(g["px0916"] > 0) - 1,
            "O_lim1": lim1,
            "O_lim5": lim5,
        }[STAT]
    g["signal"] = np.asarray(sig, dtype=float)
    g = g[g["d"] >= pd.Timestamp(pd.to_datetime(start_date).normalize())]
    rows = []
    for t in SLOTS:
        s = g[["d", "instrument", "signal"]].copy()
        s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
        rows.append(s[["date", "instrument", "signal"]])
    df = pd.concat(rows, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
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
