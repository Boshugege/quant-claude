def main(datasources, start_date, end_date):
    """
    快速版多源自适应 IC 加权复合（33 号的性能优化版）。

    与 33 号的区别（性能）：
      - 不用窗口函数 LAG，全程只有一次流式 GROUP BY，不需要对全表排序，内存不随数据量膨胀；
      - 只读 25 个字段（33 号读 38 个），去掉了需要十个价格字段的“价距密度”分量；
      - 成交方向改用分钟内收盘位置 CLV（无需上一分钟盘口）。
    信息源：
      book : 逐档挂单量不平衡 + 逐档委托笔数不平衡（截面排名后合成）
      flow : 分钟资金流 CMF = Σ volume * (2*close - high - low)/(high - low) / Σ volume
      rev  : 30 分钟 VWAP 偏离 ln(VWAP / 段末收盘价)
    合成：截面居中排名，按“历史已实现截面 Rank IC 的指数加权均值”加权；
    IC 序列整体后移一个截面后才使用，不含任何未来信息；预热期向先验（只有 book 为正）收缩。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    # ---------- 可调参数 ----------
    # 已实现收益口径："close" = 下一段 VWAP / 本段末收盘价 - 1；"vwap" = 下一段 VWAP / 本段 VWAP - 1
    TARGET_MODE = "close"
    IC_HALFLIFE = 96
    PRIOR_STRENGTH = 24
    MIN_STOCKS = 30
    PRIOR = {"c_book": 0.08, "c_flow": 0.0, "c_rev": 0.0}

    bar1m = datasources["bar1m"]
    levels = range(1, 6)

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_volume{i}, 0) AS DOUBLE) AS bv{i}" for i in levels]
        + [f"CAST(COALESCE(ask_volume{i}, 0) AS DOUBLE) AS av{i}" for i in levels]
        + [f"CAST(COALESCE(bid_num_orders{i}, 0) AS DOUBLE) AS bn{i}" for i in levels]
        + [f"CAST(COALESCE(ask_num_orders{i}, 0) AS DOUBLE) AS an{i}" for i in levels]
    )
    vol_terms = " + ".join(f"(bv{i} - av{i}) / (bv{i} + av{i} + 1e-8)" for i in levels)
    cnt_terms = " + ".join(f"(bn{i} - an{i}) / (bn{i} + an{i} + 1e-8)" for i in levels)

    sql = f"""
    WITH raw AS (
        SELECT
            instrument_id,
            CAST(CAST(date AS TIMESTAMP) AS DATE) AS trading_day,
            hour(date) * 10000 + minute(date) * 100 + second(date) AS hms,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px,
            CAST(COALESCE(high, 0) AS DOUBLE) AS hi,
            CAST(COALESCE(low, 0) AS DOUBLE) AS lo,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,
            {raw_cols}
        FROM {bar1m}
    ),
    seg AS (
        SELECT
            *,
            CASE
                WHEN hms > 90000  AND hms <= 93000  THEN 93000
                WHEN hms > 93000  AND hms <= 100000 THEN 100000
                WHEN hms > 100000 AND hms <= 103000 THEN 103000
                WHEN hms > 103000 AND hms <= 110000 THEN 110000
                WHEN hms > 110000 AND hms <= 113000 THEN 113000
                WHEN hms > 130000 AND hms <= 133000 THEN 133000
                WHEN hms > 133000 AND hms <= 140000 THEN 140000
                WHEN hms > 140000 AND hms <= 143000 THEN 143000
                WHEN hms > 143000 AND hms <= 150000 THEN 150000
                ELSE -1
            END AS time_segment
        FROM raw
    ),
    snap AS (
        SELECT
            trading_day, time_segment, instrument_id, hms, vol, amt,
            CASE WHEN px > 0 THEN px END AS px,
            CASE WHEN bv1 > 0 AND av1 > 0 THEN ({vol_terms}) / 5.0 END AS s_vol,
            CASE WHEN bv1 > 0 AND av1 > 0 THEN ({cnt_terms}) / 5.0 END AS s_cnt,
            CASE WHEN hi > lo AND vol > 0 THEN (2.0 * px - hi - lo) / (hi - lo) END AS clv
        FROM seg
        WHERE time_segment > 0
    ),
    win AS (
        SELECT
            trading_day, time_segment, instrument_id,
            AVG(s_vol) AS f_vol,
            AVG(s_cnt) AS f_cnt,
            SUM(clv * vol) / NULLIF(SUM(CASE WHEN clv IS NOT NULL THEN vol END), 0) AS flow,
            SUM(amt) / NULLIF(SUM(vol), 0) AS vwap,
            arg_max(px, CASE WHEN px IS NOT NULL THEN hms END) AS last_px
        FROM snap
        GROUP BY trading_day, time_segment, instrument_id
    )
    SELECT w.trading_day, w.time_segment, i.instrument,
           w.f_vol, w.f_cnt, w.flow, w.vwap, w.last_px
    FROM win w
    LEFT JOIN all_instruments i USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    df["instrument"] = df["instrument"].astype(str)
    df["time_segment"] = df["time_segment"].astype(int)
    df["trading_day"] = pd.to_datetime(df["trading_day"])
    num_cols = ["f_vol", "f_cnt", "flow", "vwap", "last_px"]
    df[num_cols] = df[num_cols].astype("float64").replace([np.inf, -np.inf], np.nan)

    px_ok = (df["vwap"] > 0) & (df["last_px"] > 0)
    df["rev"] = np.where(px_ok, np.log(df["vwap"].where(px_ok, 1.0) / df["last_px"].where(px_ok, 1.0)), np.nan)

    # ---------- 已实现收益（只用于计算历史 IC） ----------
    next_seg = {93000: 100000, 100000: 103000, 103000: 110000, 110000: 113000,
                113000: 133000, 133000: 140000, 140000: 143000, 143000: 150000}
    df = df.sort_values(["instrument", "trading_day", "time_segment"]).reset_index(drop=True)
    grp = df.groupby(["instrument", "trading_day"], sort=False)
    nseg = grp["time_segment"].shift(-1)
    nvwap = grp["vwap"].shift(-1)
    base = df["last_px"] if TARGET_MODE == "close" else df["vwap"]
    ok_fwd = (nseg == df["time_segment"].map(next_seg)) & (base > 0) & (nvwap > 0)
    df["fwd"] = np.where(ok_fwd, nvwap / base.where(ok_fwd, 1.0) - 1.0, np.nan)
    df = df[df["time_segment"] != 150000].copy()

    seg = df["time_segment"]
    df["date"] = (
        df["trading_day"]
        + pd.to_timedelta(seg // 10000, unit="h")
        + pd.to_timedelta(seg // 100 % 100, unit="m")
    )

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    stk_pool["date"] = pd.to_datetime(stk_pool["date"])
    stk_pool["instrument"] = stk_pool["instrument"].astype(str)

    keep = ["date", "instrument", "f_vol", "f_cnt", "flow", "rev", "fwd"]
    res = pd.merge(df[keep], stk_pool, how="inner", on=["date", "instrument"])
    res = res.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)

    # ---------- 截面居中排名 ----------
    by_date = res.groupby("date")
    raw_ranks = by_date[["f_vol", "f_cnt", "flow", "rev"]].rank(pct=True) - 0.5
    res["book_raw"] = raw_ranks[["f_vol", "f_cnt"]].mean(axis=1, skipna=True)
    res["c_book"] = res.groupby("date")["book_raw"].rank(pct=True) - 0.5
    res["c_flow"] = raw_ranks["flow"]
    res["c_rev"] = raw_ranks["rev"]
    res["y"] = by_date["fwd"].rank(pct=True) - 0.5
    comps = ["c_book", "c_flow", "c_rev"]

    # ---------- 每个截面的 Rank IC ----------
    dates = pd.Index(np.sort(res["date"].unique()))
    ic = pd.DataFrame(index=dates, columns=comps, dtype="float64")
    for c in comps:
        m = res[c].notna() & res["y"].notna()
        t = pd.DataFrame({"date": res["date"][m], "x": res[c][m], "y": res["y"][m]})
        t["x"] = t["x"] - t.groupby("date")["x"].transform("mean")
        t["y"] = t["y"] - t.groupby("date")["y"].transform("mean")
        t["xy"] = t["x"] * t["y"]
        t["xx"] = t["x"] * t["x"]
        t["yy"] = t["y"] * t["y"]
        s = t.groupby("date").agg(xy=("xy", "sum"), xx=("xx", "sum"), yy=("yy", "sum"), n=("xy", "size"))
        val = s["xy"] / np.sqrt(s["xx"] * s["yy"])
        val[(s["n"] < MIN_STOCKS) | ~np.isfinite(val)] = np.nan
        ic[c] = val.reindex(dates)

    # ---------- 只用过去 IC：整体后移一个截面，再指数加权，并向先验收缩 ----------
    hist = ic.shift(1)
    ewm = hist.ewm(halflife=IC_HALFLIFE, ignore_na=True).mean().ffill()
    n_obs = hist.notna().cumsum()
    shrink = n_obs / (n_obs + PRIOR_STRENGTH)
    weights = pd.DataFrame(index=dates, columns=comps, dtype="float64")
    for c in comps:
        weights[c] = shrink[c] * ewm[c].fillna(0.0) + (1.0 - shrink[c]) * PRIOR[c]

    w = weights.reindex(res["date"]).to_numpy()
    x = res[comps].to_numpy(dtype="float64")
    all_nan = np.isnan(x).all(axis=1)
    res["factor"] = np.where(all_nan, np.nan, np.nansum(np.nan_to_num(x) * w, axis=1))
    res["factor"] = res["factor"].replace([np.inf, -np.inf], np.nan)

    return res[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


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
