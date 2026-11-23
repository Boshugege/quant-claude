def main(datasources, start_date, end_date):
    """
    平台原始 IQR 结构（同伴残差 + 四特征中性化 + IQR Winsor 均值）的首窗去重四网络中位数版本。

    数据读取、窗口末价格和底层信号完全沿用平台高分
    jyc_peer_component_iqr_winsor_v1（0.80917），不纳入 09:25 竞价快照，也不做窗口末价格兜底。
    在 138 号（首窗去重 + 原始 / 超额 x 日频 / 30 分钟四张同伴网络，16 列 IQR）基础上只改一处：
      同伴基准由等权均值改为中位数（132 号已验证的无参稳健化），单只同伴的异常涨跌不再拖动整组基准。
    138 的两个无参结构改动保持不变：
      1. 首窗去重：当日首个窗口的 30 分钟反转以昨收为基准，与 r_day 完全相同，首窗把 r_30 置为缺失，避免重复计票；
      2. 四张同伴网络：超额网络先把每个收益观测减去同一时点的截面均值，再按相关性取前 30 名同伴；
         每个分量在每张网络上各做一次同伴中位数残差与四特征中性化，16 列做 IQR（09:30 截面 12 列）。
    不新增任何参数，同伴数、回看期、最少观测都与原始 IQR 相同。
    本地开发 / 验证 / 保留三段（括号内依次为 138、原始 IQR）：
      IC 0.08863 / 0.07916 / 0.06639（0.08780 / 0.07828 / 0.06572；0.08636 / 0.07702 / 0.06470）；
      ICIR 1.4657 / 1.6266 / 1.1415（1.4655 / 1.6199 / 1.1361；1.4443 / 1.5882 / 1.1182）；
      Sharpe 49.37 / 50.89 / 36.40（49.13 / 50.62 / 36.17；48.54 / 49.99 / 35.71）；
      压力 1.3877 / 1.5418 / 1.0743（1.3895 / 1.5402 / 1.0632；1.3681 / 1.5036 / 1.0460）；
      换手 0.5930 / 0.5934 / 0.5895（0.5916 / 0.5923 / 0.5880；0.5896 / 0.5911 / 0.5861）。
    公共构造：
      反转分量（与 90 号相同，均只用过去数据）：
        r_30  : 30 分钟反转 ln(上一窗口末价 / 当前价)，当日首个窗口以昨收为基准（57 号）
        r_rv4 / r_rv8 : 滚动 4/8 个窗口（2/4 小时，跨日）清洗后成交 VWAP 偏离 ln(滚动 VWAP / 当前价)
        r_day : 日内反转 ln(昨收 / 当前价)（56 号）
      同伴（伪行业）残差：每个自然月月初，用此前 60 个交易日的日收益 ln(收盘段价格 / 昨收) 计算股票池内两两相关系数，
        每只股票取相关性最高的 30 只为同伴（至少 30 个共同观测；只用过去数据）。分量残差 = 分量 - 同伴中位数
        （同伴有效值不足 3 个时减截面均值）。思路：行业内反转强于全市场反转（Da, Liu & Schaumburg 2014），
        去掉“整组一起涨跌”的部分后，只保留个股相对同类股票的过度反应。
      四特征中性化：因子截面排名对 规模（-ln 最近 40 窗口成交额）、特质波动、非流动性（Amihud）、低价 四个特征的截面排名
        做逐截面回归，取残差。反转信号天然偏向小盘、高波动股票，而这些暴露本身会拖累排序；剔除后 IC、IC_IR、多空夏普同时提升。
      数据清洗：成交均价落在该分钟 [最低价 x 0.98, 最高价 x 1.02] 之外的分钟不计入任何 VWAP / 成交额累加。

    稳健性：最终缺失补 0（中性），每个截面缺失率为 0；向前取不到足够历史时同伴退化为截面均值、特征排名记为中性。
    性能：纯 SQL 向量化（单次 GROUP BY，无窗口函数、无 Python UDF）；相关矩阵每月计算一次（约 1200 x 1200），
      同伴均值按月份用数组索引批量计算。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取数据：同伴相关系数需要月初之前 60 个交易日的日收益（约 90 个自然日，加上月内偏移与长假留足余量）。
    LOOKBACK_DAYS = 150
    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS = 30, 60, 30
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
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(pre_close, 0) AS DOUBLE) AS pre_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt
        FROM {bar1m}
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            pre_close,
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
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close
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
        f.sum_amt, f.sum_vol, f.last_px, f.pre_close
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)

    # 股票池：同伴只在股票池内挑选（向前取历史区间的成分，取不到时用评估区间的成分）。
    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
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
    df["r_day"] = safe_log_ratio(df["pre_close"], px)
    # 首窗 r_30 与 r_day 相同，参与合成的分量在首窗置缺失，避免重复计票。
    df["r_30d"] = df["r_30"].where(same_day)

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
    df["c_max"] = -roll(win_ret, 40, "max", 10)

    # 同伴：每个自然月月初，用此前 PEER_LOOKBACK 个交易日的收益计算相关系数，两种频率各建一张网络：
    #   日频：日收益（收盘段价格 / 昨收），至少 PEER_MIN_OBS 个共同观测；
    #   30 分钟：窗口收益（上一窗口末价，当日首个窗口用昨收），取回看期内全部窗口，至少 4 x PEER_MIN_OBS 个共同观测。
    daily = df.groupby(["day", "instrument"], sort=False).agg(px_last=("last_px", "last"), pc=("pre_close", "first"))
    daily_ret = safe_log_ratio(daily["px_last"], daily["pc"]).unstack("instrument").sort_index()
    peer_insts = daily_ret.columns.values
    all_days = daily_ret.index
    day_month = all_days.to_period("M")
    out_months = pd.to_datetime(stk_pool["date"]).dt.normalize().dt.to_period("M").unique()
    win_ret30 = pd.Series(
        (-df["r_30"]).values, index=pd.MultiIndex.from_arrays([df["date"].values, df["instrument"].values])
    ).unstack().sort_index().reindex(columns=peer_insts)

    def build_peers(ret_wide, row_day, need):
        """返回 {月份: 同伴下标矩阵}；ret_wide 每行是一个收益观测，row_day 为该行所属交易日。"""
        peers = {}
        for m in out_months:
            in_month = all_days[day_month == m]
            if len(in_month) == 0:
                continue
            prior = all_days[all_days < in_month[0]][-PEER_LOOKBACK:]
            if len(prior) < PEER_MIN_OBS:
                continue
            hist = ret_wide[(row_day < in_month[0]) & (row_day >= prior[0])]
            ok = (hist.notna().sum() >= need).values
            corr = hist.loc[:, ok].corr(min_periods=need).values
            np.fill_diagonal(corr, -np.inf)
            corr = np.nan_to_num(corr, nan=-np.inf)
            sub = np.where(ok)[0]
            idx = np.full((len(peer_insts), PEER_K), -1, dtype=np.int64)
            idx[sub] = sub[np.argsort(-corr, axis=1, kind="stable")[:, :PEER_K]]
            peers[m] = idx
        return peers

    # 超额网络：每个收益观测先减去同一时点的截面均值，再按超额收益的相关性挑同伴（去掉市场共振带来的伪同伴）。
    peers_by_freq = {
        "day": build_peers(daily_ret, all_days, PEER_MIN_OBS),
        "30": build_peers(win_ret30, win_ret30.index.normalize(), PEER_MIN_OBS * 4),
        "day_excess": build_peers(daily_ret.sub(daily_ret.mean(axis=1), axis=0), all_days, PEER_MIN_OBS),
        "30_excess": build_peers(win_ret30.sub(win_ret30.mean(axis=1), axis=0), win_ret30.index.normalize(), PEER_MIN_OBS * 4),
    }

    df = df.replace([np.inf, -np.inf], np.nan)
    keep = ["r_30d", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max"]
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool[["date", "instrument"]], how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)
    result_month = result["date"].dt.to_period("M")

    def peer_median(col, peers_by_month):
        """每个截面上同伴的 col 中位数（至少 3 个有效值），没有同伴信息时为缺失。"""
        out = np.full(len(result), np.nan)
        for m, rows in result.groupby(result_month, sort=False).indices.items():
            idx = peers_by_month.get(m)
            if idx is None:
                continue
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            valid = idx >= 0
            g = wide.values[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            cnt = (~np.isnan(g)).sum(axis=2)
            pm = np.where(cnt >= 3, np.nanmedian(g, axis=2), np.nan)
            r_pos = wide.index.get_indexer(sub["date"])
            c_pos = pd.Index(peer_insts).get_indexer(sub["instrument"])
            out[rows] = np.where(c_pos >= 0, pm[r_pos, np.maximum(c_pos, 0)], np.nan)
        return pd.Series(out, index=result.index)

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

    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    neutral_cols = []
    for c in ["r_30d", "r_rv4", "r_rv8", "r_day"]:
        cross_mean = result.groupby("date")[c].transform("mean")
        for f, peers in peers_by_freq.items():
            # 每个分量 x 每张同伴网络：同伴中位数残差后做四特征中性化。
            result[f"{c}_p{f}"] = (result[c] - peer_median(c, peers).fillna(cross_mean)).where(result[c].notna())
            name = f"{c}_n{f}"
            result[name] = neutralize(f"{c}_p{f}", CHARS4)
            neutral_cols.append(name)
    # IQR Winsor 均值：逐行把各列截到该行 [Q1, Q3] 再取均值（与平台原始 IQR 相同）。
    neutral_ranks = result.groupby("date")[neutral_cols].rank(pct=True) - 0.5
    lower = neutral_ranks.quantile(0.25, axis=1)
    upper = neutral_ranks.quantile(0.75, axis=1)
    result["factor"] = neutral_ranks.clip(lower=lower, upper=upper, axis=0).mean(axis=1, skipna=True)
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)
