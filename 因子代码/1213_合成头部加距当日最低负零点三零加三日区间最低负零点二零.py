def main(datasources, start_date, end_date):
    """【新根因子第二轮·双新根合成 1213】头部 490 + 1133（距当日最低幅度）×-0.20 + 1200（距 3 日区间最低幅度）×W2

    两个新根各自先跑出最终因子（`_root_impl` = 1133、`_root_impl_b` = 1200，均逐字节复用）再逐截面取 pct 秩求和：
    B = rank(头部) + (-0.30)·rank(1133) + W2·rank(1200)。平台口径：只保留 8 个约定截面、股票池骨架、缺失填 0、0930 置 0。
    实测（评分机 2.5.3，基线 490；ΔS=3·ΔIC-0.5·Δ换手+0.2·ΔIC_IR+0.008·Δ多空）：
      本配置（1133×-0.30 + 1200×-0.20）实测 ΔS +0.0264、ΔIC +0.00106、ΔIC_IR -0.0054、Δ换手 -0.0423、压力 1.389，
      是第二轮 ΔS 与 IC_IR 兼顾最好的一档。
    """
    import numpy as np
    import pandas as pd
    import dai

    W2 = -0.20

    _head = _head_impl(datasources, start_date, end_date).rename(columns={"factor": "fh"})
    _ra = _root_impl(datasources, start_date, end_date).rename(columns={"factor": "fa"})
    _rb = _root_impl_b(datasources, start_date, end_date).rename(columns={"factor": "fb"})
    for _f in (_head, _ra, _rb):
        _f["date"] = pd.to_datetime(_f["date"])
        _f["instrument"] = _f["instrument"].astype(str)
    _m = _head.merge(_ra, on=["date", "instrument"], how="inner").merge(_rb, on=["date", "instrument"], how="inner")
    _m = _m[np.isfinite(_m["fh"]) & np.isfinite(_m["fa"]) & np.isfinite(_m["fb"])].copy()
    _m["rh"] = _m.groupby("date")["fh"].rank(pct=True) - 0.5
    _m["ra"] = _m.groupby("date")["fa"].rank(pct=True) - 0.5
    _m["rb"] = _m.groupby("date")["fb"].rank(pct=True) - 0.5
    _m["factor"] = _m["rh"].to_numpy() + -0.30 * _m["ra"].to_numpy() + W2 * _m["rb"].to_numpy()

    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)].drop_duplicates(["date", "instrument"])
    _out = _pool.merge(_m[["date", "instrument", "factor"]], on=["date", "instrument"], how="left")
    _out["factor"] = _out["factor"].astype("float64").fillna(0.0)
    _out.loc[_out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return _out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def _head_impl(datasources, start_date, end_date):
    """长窗口软同伴（240 日）+ 六特征 1.5 倍中性化（新增换手率与市值二次项）+（09:30 置零版）。

    148（换手率特征，0.834）与 149（市值二次项，0.834）各自与底座持平，这里同时放进中性化特征集，
      检验两个「官方风格未覆盖的暴露」一起剔除后能否超过 144。
    09:30 修复：同 104 号对照版本。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取数据：同伴相关系数需要月初之前 60 个交易日的日收益（约 90 个自然日，加上月内偏移与长假留足余量）。
    LOOKBACK_DAYS = 400
    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 240, 120, 2.0
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,
            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals
        FROM {bar1m}
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            pre_close,
            px_high,
            px_low,
            deals,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN amt ELSE 0
            END AS amt_c,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN vol ELSE 0
            END AS vol_c,
            CASE
                WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - px_low ELSE 0 END AS rng_c,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,
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
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt_c) AS sum_amt,
            SUM(vol_c) AS sum_vol,
            SUM(body_c) AS sum_body,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low,
            SUM(rng_c) AS sum_rng,
            SUM(upsh_c) AS sum_upsh,
            SUM(deals) AS sum_deals
        FROM cte_snapshot
        WHERE time_segment != -1
        GROUP BY instrument_id, trading_day, time_segment
    )
    SELECT
        CAST(CONCAT(
            f.trading_day,
            ' ',
            strftime(strptime(LPAD(f.time_segment, 6, '0'), '%H%M%S'), '%H:%M:%S')
        ) AS DATETIME) AS date,
        all_instruments.instrument,
        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)

    # 股票池：同伴只在股票池内挑选（向前取历史区间的成分，取不到时用评估区间的成分）。
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")
    pool_hist = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [query_start, end_date]},
        compression=True,
    ).df()
    pool_hist["date"] = pd.to_datetime(pool_hist["date"])
    pool_hist["instrument"] = pool_hist["instrument"].astype(str)
    stk_pool = pool_hist[(pool_hist["date"] >= pd.to_datetime(start_date)) & (pool_hist["date"] <= pd.to_datetime(end_date))]
    universe = set(pool_hist["instrument"].unique())
    df = df[df["instrument"].isin(universe)]
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()

    # 官方暴露表（日频）里的流通市值：用于构造「官方风格没有覆盖」的中性化特征。
    _exp = dai.query(
        "SELECT date, instrument, float_market_cap FROM %s" % exposure_table,
        filters={"date": [query_start, end_date]}, compression=True,
    ).df()
    _exp["day"] = pd.to_datetime(_exp["date"]).dt.normalize()
    _exp["instrument"] = _exp["instrument"].astype(str)
    _mcap = _exp.groupby(["instrument", "day"], sort=False)["float_market_cap"].last()
    # 暴露表第 t 天的流通市值按当天 15:00 收盘价计算，盘中截面不可得：一律取该股票前一交易日的值。
    _mcap = _mcap.sort_index().groupby(level=0).shift(1)
    _midx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["mcap"] = pd.Series(_mcap.reindex(_midx).to_numpy(), index=df.index)
    df["mcap"] = df["mcap"].where(df["mcap"] > 0)
    # 官方 10 风格暴露 + 行业码（与平台标签残差化口径一致），用于因子自身的中性化测试。
    _st = dai.query(
        "SELECT date, instrument, SIZE, BETA, MOMENTUM, RESVOL, SIZENL, BTOP, LIQUIDTY, EARNYILD, GROWTH, LEVERAGE, industry_level1_code FROM %s" % exposure_table,
        filters={"date": [query_start, end_date]}, compression=True,
    ).df()
    _st["day"] = pd.to_datetime(_st["date"]).dt.normalize()
    _st["instrument"] = _st["instrument"].astype(str)
    _st = _st.drop_duplicates(["instrument", "day"])
    # 风格暴露与行业同样取前一交易日的值（第 t 天的暴露在当天收盘后才可得）。
    _st = _st.sort_values(["instrument", "day"]).reset_index(drop=True)
    for _c in ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE", "industry_level1_code"]:
        _st[_c] = _st.groupby("instrument")[_c].shift(1)
    _st = _st.dropna(subset=["industry_level1_code"])
    _sg = _st.set_index(["instrument", "day"])
    _sid = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    for _c in ['SIZE', 'BETA', 'MOMENTUM', 'RESVOL', 'SIZENL', 'BTOP', 'LIQUIDTY', 'EARNYILD', 'GROWTH', 'LEVERAGE']:
        df["s_" + _c] = pd.Series(_sg[_c].reindex(_sid).to_numpy(), index=df.index)
    _ind = _sg["industry_level1_code"].astype(str)
    for _c in sorted(_ind.dropna().unique()):
        df["i_" + _c] = (_ind.reindex(_sid).to_numpy() == _c).astype(float)

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    def vwap_of(amt, vol):
        return (amt / vol.where(vol > 0)).where(amt > 0)

    def roll(s, k, fn, min_periods):
        return s.groupby(df["instrument"], sort=False).transform(lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())

    grp = df.groupby("instrument", sort=False)
    grp_day = df.groupby(["instrument", "day"], sort=False)
    same_day = grp["day"].shift(1) == df["day"]
    px = df["last_px"]

    day_vwap = vwap_of(grp_day["sum_amt"].cumsum(), grp_day["sum_vol"].cumsum())
    r_dvwap = safe_log_ratio(day_vwap, px)

    def rolling_vwap_rev(k):
        return safe_log_ratio(vwap_of(roll(df["sum_amt"], k, "sum", 1), roll(df["sum_vol"], k, "sum", 1)), px).fillna(r_dvwap)

    df["r_30"] = safe_log_ratio(grp["last_px"].shift(1).where(same_day, df["pre_close"]), px)
    df["r_rv4"] = rolling_vwap_rev(4)
    df["r_rv8"] = rolling_vwap_rev(8)
    df["r_rv16"] = rolling_vwap_rev(16)
    df["r_day"] = safe_log_ratio(df["pre_close"], px)

    # 个股 beta（40 窗口滚动回归，截断 [-1, 4]，不足取 1），用于特质波动。
    win_ret = safe_log_ratio(px, grp["last_px"].shift(1))
    ret30 = safe_log_ratio(px, grp["last_px"].shift(1).where(same_day)).fillna(safe_log_ratio(px, df["pre_close"]))
    mkt30 = ret30.groupby(df["date"]).transform("mean")
    ex, ey = roll(ret30, 40, "mean", 10), roll(mkt30, 40, "mean", 10)
    exy, ey2 = roll(ret30 * mkt30, 40, "mean", 10), roll(mkt30 ** 2, 40, "mean", 10)
    var_m = (ey2 - ey ** 2).where((ey2 - ey ** 2) > 1e-12)
    beta = ((exy - ex * ey) / var_m).clip(-1, 4).fillna(1.0)

    # 股票特征：规模（-ln 最近 40 窗口成交额）、特质波动（取负）、非流动性（Amihud）、低价、MAX（取负）。
    df["c_size"] = -np.log1p(roll(df["sum_amt"], 40, "sum", 10))
    idio = win_ret - beta * win_ret.groupby(df["date"]).transform("mean")
    df["c_ivol"] = -roll(idio, 40, "std", 10)
    df["c_illiq"] = roll(win_ret.abs() / (df["sum_amt"] / 1e8).where(df["sum_amt"] > 0), 40, "mean", 10)
    df["c_lowpx"] = -np.log(px.where(px > 0))

    # 换手率代理：最近 40 个窗口成交额 ÷ 流通市值（官方风格里没有直接的换手率暴露）。
    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]

    # 流通市值的非线性项：ln(流通市值) 的平方（官方 SIZE 风格只覆盖线性部分）。
    df["c_mcap2"] = np.log(df["mcap"]) ** 2
    df["c_max"] = -roll(win_ret, 40, "max", 10)
    # 分钟上影线占比：窗口内 Σ(最高-实体上沿)/Σ(最高-最低)，40 窗口均值取负（冲高回落越多，反转越强）。
    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)
    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)
    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())
    # 平均单笔成交额：Σ成交额/Σ委托笔数 的 40 窗口均值（大单/小单主导窗口的代理）。
    _avgtr = df["sum_amt"] / df["sum_deals"].where(df["sum_deals"] > 0)
    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)

    # 同伴：每个自然月月初，用此前 PEER_LOOKBACK 个交易日的日收益（收盘段价格 / 昨收）计算相关系数。
    daily = df.groupby(["day", "instrument"], sort=False).agg(px_last=("last_px", "last"), pc=("pre_close", "first"))
    daily_ret = safe_log_ratio(daily["px_last"], daily["pc"]).unstack("instrument").sort_index()
    peer_insts = daily_ret.columns.values
    daily_amt = df.groupby(["day", "instrument"], sort=False)["sum_amt"].sum().unstack("instrument").reindex(columns=peer_insts).sort_index()
    all_days = daily_ret.index
    day_month = all_days.to_period("M")
    peers_by_month = {}
    out_months = pd.to_datetime(stk_pool["date"]).dt.normalize().dt.to_period("M").unique()
    for m in out_months:
        in_month = all_days[day_month == m]
        if len(in_month) == 0:
            continue
        hist = daily_ret.loc[daily_ret.index < in_month[0]].tail(PEER_LOOKBACK)
        if len(hist) < PEER_MIN_OBS:
            continue
        ok = (hist.notna().sum() >= PEER_MIN_OBS).values
        corr = hist.loc[:, ok].corr(min_periods=PEER_MIN_OBS).values
        np.fill_diagonal(corr, np.nan)
        corr = np.nan_to_num(corr, nan=-1.0)          # 缺失相关性记 -1（等价于不选）
        sub = np.where(ok)[0]
        top = np.argsort(-corr, axis=1, kind="stable")[:, :PEER_K]
        idx = np.full((len(peer_insts), PEER_K), -1, dtype=np.int64)
        idx[sub] = sub[top]
        cvals = corr[np.arange(len(corr))[:, None], top]
        # 规模加权：相关度^p × 该同伴前 PEER_LOOKBACK 日的平均成交额（大票的共同运动信息权重更高）。
        _aw = daily_amt.loc[daily_amt.index < in_month[0]].tail(PEER_LOOKBACK).mean().fillna(0.0).values
        _aw = np.where(_aw > 0, _aw, 0.0)
        wts = (np.clip(cvals, 0.0, None) ** PEER_POWER) * _aw[sub][:, None]
        wsum = wts.sum(axis=1, keepdims=True)
        wts = np.where(wsum > 0, wts / np.where(wsum > 0, wsum, 1.0), 0.0)
        full_w = np.zeros((len(peer_insts), PEER_K))
        full_w[sub] = wts
        peers_by_month[m] = (idx, full_w)

    df = df.replace([np.inf, -np.inf], np.nan)
    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]
    keep = keep + sorted([c for c in df.columns if c.startswith("i_")])
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool[["date", "instrument"]], how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)
    result_month = result["date"].dt.to_period("M")

    def peer_mean(col):
        """每个截面上同伴的 col 均值（至少 3 个有效值），没有同伴信息时为缺失。"""
        out = np.full(len(result), np.nan)
        for m, rows in result.groupby(result_month, sort=False).indices.items():
            rec = peers_by_month.get(m)
            if rec is None:
                continue
            idx, wts = rec
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            valid = idx >= 0
            g = wide.values[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            ww = np.where(valid[None, :, :], wts[None, :, :], 0.0)
            okm = ~np.isnan(g)
            cnt = okm.sum(axis=2)
            wsum = (ww * okm).sum(axis=2)
            pm = np.where((cnt >= 3) & (wsum > 0), np.nansum(np.where(okm, g, 0.0) * ww, axis=2) / np.where(wsum > 0, wsum, 1.0), np.nan)
            r_pos = wide.index.get_indexer(sub["date"])
            c_pos = pd.Index(peer_insts).get_indexer(sub["instrument"])
            out[rows] = np.where(c_pos >= 0, pm[r_pos, np.maximum(c_pos, 0)], np.nan)
        return pd.Series(out, index=result.index)

    def rank_blend(cols, weights=None):
        """各分量在股票池内逐截面居中排名后加权平均（缺失分量不参与，权重按可用分量重新归一）。"""
        weights = weights or [1.0] * len(cols)
        ranks = (result.groupby("date")[cols].rank(pct=True) - 0.5).to_numpy()
        w = np.array(weights, dtype=float)
        has = ~np.isnan(ranks)
        den = (has * w).sum(axis=1)
        num = np.nansum(np.nan_to_num(ranks) * w, axis=1)
        return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=result.index)

    def neutralize(col, chars, frac=1.0):
        """截面中性化：col 的居中排名对各特征居中排名（缺失记 0）逐截面 OLS（含截距），返回 排名 - frac x 拟合值。"""
        y = result.groupby("date")[col].rank(pct=True) - 0.5
        X = np.column_stack([(result.groupby("date")[c].rank(pct=True) - 0.5).fillna(0.0).values for c in chars])
        ok = y.notna().values
        codes, uniq = pd.factorize(result["date"].values)
        k, n_grp = X.shape[1], len(uniq)
        Xc = np.column_stack([np.ones(len(result)), X]) * ok[:, None]
        yv = np.nan_to_num(y.values) * ok
        xtx = np.zeros((n_grp, k + 1, k + 1))
        xty = np.zeros((n_grp, k + 1))
        for i in range(k + 1):
            xty[:, i] = np.bincount(codes, Xc[:, i] * yv, minlength=n_grp)
            for j in range(i, k + 1):
                v = np.bincount(codes, Xc[:, i] * Xc[:, j], minlength=n_grp)
                xtx[:, i, j] = v
                xtx[:, j, i] = v
        b = np.linalg.solve(xtx + 1e-9 * np.eye(k + 1), xty[..., None])[..., 0]
        fit = (X * b[codes, 1:]).sum(axis=1)
        return pd.Series(np.where(ok, y.values - frac * fit, np.nan), index=result.index)

    def peer_ladder():
        cols = []
        for c in ["r_30", "r_rv4", "r_rv8", "r_day"]:
            result[c + "_p"] = result[c] - peer_mean(c).fillna(result.groupby("date")[c].transform("mean"))
            cols.append(c + "_p")
        return rank_blend(cols)

    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"] + sorted([c for c in df.columns if c.startswith("i_")])
    result["ladder"] = peer_ladder()
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])
    result["factor"] = neutralize("signal", CHARS4, 2.5)
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面。
    result.loc[result["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


def _root_impl(datasources, start_date, end_date):
    """【新根因子 方向F：因果日内价格位置】当日截至当前窗口的相对位置与对当日累计 VWAP 的偏离。
    【方向F变体 1133：距当日最低幅度】

    假设：日内相对位置（现价在当日已走出的高低区间里的位置）与"相对当日累计成交均价 VWAP 的偏离"
    刻画当天的买卖压力与套牢/浮盈结构，对之后 30 分钟收益有预测力。

    全部**只用当日截至当前窗口**的信息：SQL 按段给末价、段内最高/最低、段内成交额与成交量，
    pandas 里做当日累计最高/最低/VWAP（cummax/cummin/cumsum，严格因果）。不读 vwap 表。
    不含反转阶梯、K 线形态、泄漏分量。

    参数：STAT=pos_range|dev_vwap|dev_vwap_norm|dist_low|dist_high|blend。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "dist_low"
    bar1m = datasources["bar1m"]
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=10)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(b.high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(b.low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close,
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
        LEFT JOIN all_instruments ai USING (instrument_id)
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
        MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                  OR (time_segment = 93000 AND hms = '092500') THEN px_close ELSE 0 END) AS last_px,
        MAX(px_high) AS seg_high,
        MIN(CASE WHEN px_low > 0 THEN px_low END) AS seg_low,
        SUM(amt) AS seg_amt,
        SUM(vol) AS seg_vol,
        MAX(pre_close) AS pre_close
    FROM cte_raw
    WHERE time_segment != -1
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
    df["cum_high"] = df.groupby(keys, sort=False)["seg_high"].cummax()
    df["cum_low"] = df.groupby(keys, sort=False)["seg_low"].cummin()
    df["cum_amt"] = df.groupby(keys, sort=False)["seg_amt"].cumsum()
    df["cum_vol"] = df.groupby(keys, sort=False)["seg_vol"].cumsum()
    px = df["last_px"].where(df["last_px"] > 0)
    pre = df["pre_close"].where(df["pre_close"] > 0)
    vwap = (df["cum_amt"] / df["cum_vol"].where(df["cum_vol"] > 0)).where(df["cum_amt"] > 0)
    rng = (df["cum_high"] - df["cum_low"])
    pos_range = (px - df["cum_low"]) / rng.where(rng > 0)
    dev_vwap = (px - vwap) / pre
    dev_norm = (px - vwap) / rng.where(rng > 0)

    if STAT == "pos_range":
        df["signal"] = pos_range
    elif STAT == "dev_vwap":
        df["signal"] = dev_vwap
    elif STAT == "dev_vwap_norm":
        df["signal"] = dev_norm
    elif STAT == "dist_low":
        df["signal"] = (px - df["cum_low"]) / pre
    elif STAT == "dist_high":
        df["signal"] = (px - df["cum_high"]) / pre
    else:
        df["signal"] = np.nan
        df["_p"] = pos_range
        df["_d"] = dev_vwap
    df["signal"] = pd.Series(df["signal"], index=df.index).replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(["date", "instrument_id"])

    out = df.dropna(subset=["instrument", "signal"])[["date", "instrument", "signal"]]
    out = out.drop_duplicates(["date", "instrument"]).copy()
    if STAT == "blend":
        r1 = df.groupby("date")["_p"].rank(pct=True) - 0.5
        r2 = df.groupby("date")["_d"].rank(pct=True) - 0.5
        out = df[["date", "instrument"]].copy()
        out["factor"] = (r1.to_numpy() + r2.to_numpy()) / 2.0
    else:
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


def _root_impl_b(datasources, start_date, end_date):
    """【新根因子 第二轮 方向F深挖 1200 起】把"价格在区间里的位置"从当日扩到多日，
    并加上相对今开／昨收的偏离。全部只用**截面时刻及之前**的信息：当日累计用 cummax/cummin/cumsum，
    多日窗口只用**已收盘的整日**高低（前一日及更早），严格因果。不读 vwap 表、不读暴露表。

    参数：STAT=dist_low_m3|dist_low_m5|pos_range_m3|pos_range_m5|dev_open|day_ret。
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "dist_low_m3"
    WIN = {"dist_low_m3": 3, "dist_low_m5": 5, "pos_range_m3": 3, "pos_range_m5": 5,
           "dev_open": 1, "day_ret": 1}[STAT]
    bar1m = datasources["bar1m"]
    # 多日窗口要预留足够的暖机天数（5 日窗口 + 非交易日冗余）
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=25)).strftime("%Y-%m-%d %H:%M:%S")
    sql = f"""
    WITH cte_raw AS (
        SELECT
            b.instrument_id,
            ai.instrument,
            strftime(b.date, '%Y-%m-%d') AS trading_day,
            strftime(b.date, '%H%M%S') AS hms,
            CAST(COALESCE(b.close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(b.high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(b.low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(b.pre_close, 0) AS DOUBLE) AS pre_close,
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
        LEFT JOIN all_instruments ai USING (instrument_id)
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
        MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                  OR (time_segment = 93000 AND hms = '092500') THEN px_close ELSE 0 END) AS last_px,
        MAX(px_high) AS seg_high,
        MIN(CASE WHEN px_low > 0 THEN px_low END) AS seg_low,
        SUM(amt) AS seg_amt,
        SUM(vol) AS seg_vol,
        MAX(pre_close) AS pre_close
    FROM cte_raw
    WHERE time_segment != -1
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
    df["cum_high"] = df.groupby(keys, sort=False)["seg_high"].cummax()
    df["cum_low"] = df.groupby(keys, sort=False)["seg_low"].cummin()
    px = df["last_px"].where(df["last_px"] > 0)
    pre = df["pre_close"].where(df["pre_close"] > 0)
    # 今开：当日 09:25 集合竞价结束价（当日第一段的 last_px），只用截面时刻之前的信息
    day_open = df.loc[df["time_segment"] == 93000, ["instrument_id", "trading_day", "last_px"]]
    day_open = day_open.rename(columns={"last_px": "day_open"})
    df = df.merge(day_open, on=["instrument_id", "trading_day"], how="left")
    df["day_open"] = df["day_open"].where(df["day_open"] > 0)

    if STAT in ("dist_low_m3", "dist_low_m5", "pos_range_m3", "pos_range_m5"):
        # 先按 股票×整日 汇总出**已收盘**的整日高低（当日的高只能在收盘后算完，所以当日不用它）
        d = (df.groupby(["instrument_id", "trading_day"], sort=False)
               .agg(day_high=("seg_high", "max"), day_low=("seg_low", "min"))
               .reset_index().sort_values(["instrument_id", "trading_day"]))
        g = d.groupby("instrument_id", sort=False)
        # 前 WIN-1 个**已完成**交易日的高低（shift(1) 把当日排除；rolling 只用过去）
        d["pre_high"] = g["day_high"].transform(
            lambda s: s.shift(1).rolling(WIN - 1, min_periods=1).max())
        d["pre_low"] = g["day_low"].transform(
            lambda s: s.shift(1).rolling(WIN - 1, min_periods=1).min())
        df = df.merge(d[["instrument_id", "trading_day", "pre_high", "pre_low"]],
                      on=["instrument_id", "trading_day"], how="left")
        # 显式赋值：既把类型钉死，也让 preflight 的"列不存在"静态检查能识别这两列
        df["pre_high"] = df["pre_high"].astype("float64")
        df["pre_low"] = df["pre_low"].astype("float64")
        multi_high = np.fmax(df["pre_high"].to_numpy(float), df["cum_high"].to_numpy(float))
        multi_low = np.fmin(df["pre_low"].to_numpy(float), df["cum_low"].to_numpy(float))
        rng = multi_high - multi_low
        if STAT.startswith("dist_low"):
            sig = (px.to_numpy(float) - multi_low) / pre.to_numpy(float)
        else:
            sig = (px.to_numpy(float) - multi_low) / np.where(rng > 0, rng, np.nan)
    elif STAT == "dev_open":
        sig = (px.to_numpy(float) - df["day_open"].to_numpy(float)) / pre.to_numpy(float)
    else:  # day_ret
        sig = (px.to_numpy(float) - pre.to_numpy(float)) / pre.to_numpy(float)

    df["signal"] = pd.Series(sig, index=df.index).replace([np.inf, -np.inf], np.nan)
    df = df.drop_duplicates(["date", "instrument_id"])

    out = df.dropna(subset=["instrument", "signal"])[["date", "instrument", "signal"]]
    out = out.drop_duplicates(["date", "instrument"]).copy()
    out["factor"] = out.groupby("date")["signal"].rank(pct=True) - 0.5
    out["factor"] = out["factor"].fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    # 平台骨架：只保留 8 个约定截面，以股票池 (date, instrument) 为全量骨架（信号缺失处填 0）
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


