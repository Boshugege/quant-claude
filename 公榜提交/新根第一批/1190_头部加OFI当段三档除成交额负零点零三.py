def main(datasources, start_date, end_date):
    """【新根因子第 4 级·正式合成版 1190】490 + A1104 ×-0.03（raw）

    底座：`490_自造特征加行业中性化前日市值强度二点五探针`（头部代码逐字节内联为 `_head_impl`，单独跑出来的因子与底座完全一致）。
    新根：`1104_OFI当段三档除成交额`（同样逐字节内联为 `_root_impl`）。
    合成：B_w = 头部最终因子 + -0.03 · rank(新根最终因子)（_HEAD_MODE=raw 时头部保留原始值；rank 时头部也取 pct 秩），
    与任务书第 3 级离线网格（root_eval.py --grid）度量的对象一致；只保留平台 8 个约定截面，
    以股票池为骨架、缺失填 0，0930 置 0。
    """
    import numpy as np
    import pandas as pd
    import dai

    _HEAD_MODE = "raw"
    _head = _head_impl(datasources, start_date, end_date)
    _root = _root_impl(datasources, start_date, end_date)
    _head = _head.rename(columns={"factor": "fh"})
    _root = _root.rename(columns={"factor": "fr"})
    _head["date"] = pd.to_datetime(_head["date"])
    _root["date"] = pd.to_datetime(_root["date"])
    _head["instrument"] = _head["instrument"].astype(str)
    _root["instrument"] = _root["instrument"].astype(str)
    _m = _head.merge(_root, on=["date", "instrument"], how="inner")
    _m = _m[np.isfinite(_m["fh"]) & np.isfinite(_m["fr"])].copy()
    _m["rh"] = _m["fh"] if _HEAD_MODE == "raw" else (_m.groupby("date")["fh"].rank(pct=True) - 0.5)
    _m["rr"] = _m.groupby("date")["fr"].rank(pct=True) - 0.5
    _m["factor"] = _m["rh"].to_numpy() + -0.03 * _m["rr"].to_numpy()

    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)]
    _pool = _pool.drop_duplicates(["date", "instrument"])
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
