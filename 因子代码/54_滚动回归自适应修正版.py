def main(datasources, start_date, end_date):
    """
    滚动截面回归自适应复合（笔数 / 挂单量 / VWAP 反转）—— 46 号的修正版。

    相对 46 号的修正：
      1) 已实现收益口径改为“下一段 VWAP / 本段末价格 - 1”（TARGET_MODE = "close"）。
         依据官方评分说明（IC 越大排名越高）与 39 号 0.40763 的得分，标签以截面时刻价格为基准，
         “跌破成交均价 -> 反弹”为正 IC；按“VWAP 对 VWAP”学习会把该分量的方向学反。
      2) VWAP 分量取反转方向 c_pos = ln(窗口 VWAP / 窗口末中间价)，先验为正。
      3) 向前多取 10 天数据预热回归统计量，评估期一开始即有可靠权重。

    三个分量（均已在公榜验证为强信号，且来源不同）：
      c_cnt : 远档加权委托笔数不平衡（41 号 0.41587），并与同日上一窗口半权重平滑
      c_vol : 逐档挂单量不平衡（15 号），同样平滑（37 号 0.34280）
      c_pos : VWAP 偏离反转 ln(窗口 VWAP / 窗口末中间价)（39 号，0.40763）
    每个截面在股票池内转为居中排名 X，权重 beta 由“历史截面回归”得到：
      对每个过去的截面 s，用已实现收益 y_s（下一段 VWAP / 本段 VWAP - 1，截面排名）
      累计 A_s = X'X / n、b_s = X'y / n；
      截面 t 只使用 s < t 的统计量（整体后移一个截面），按半衰期指数加权平均，
      beta_t = (A + tau * D)^(-1) (b + tau * D * beta0)，tau 随历史样本增多而衰减（向先验收缩）。
    与简单 IC 加权相比，回归会自动处理分量之间的相关性（笔数与挂单量高度相关），
    并由数据自己决定每个分量的方向和大小。

    防未来函数：截面 s 的已实现收益在下一个截面时刻完全实现，统计量整体后移一个截面再使用；
    已做本地截断测试（截断前后因子值完全一致）。
    先验 beta0：笔数与 VWAP 位置各 0.3，挂单量 0（预热期近似 44 号的等权合成）。

    稳健性：所有分钟信号永不为空；缺失分量按 0（截面中位）处理；最终缺失补 0。
    性能：纯 SQL 向量化，读取 25 个字段，单次 GROUP BY，无窗口函数、无 Python UDF；
    跨窗口运算均在 pandas 中对已聚合的小表完成。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import math
    import numpy as np
    import pandas as pd
    import dai

    # ---------- 可调参数 ----------
    # 已实现收益口径："vwap" = 下一段 VWAP / 本段 VWAP - 1；"close" = 下一段 VWAP / 本段末价格 - 1
    TARGET_MODE = "close"
    PREV_WEIGHT = 0.5         # 笔数、挂单量分量的跨窗平滑权重
    HALFLIFE = 96             # 回归统计量指数加权半衰期（截面数，约 12 个交易日）
    PRIOR_STRENGTH = 24       # 先验相当于多少个截面的样本（约 3 个交易日）
    MIN_STOCKS = 30
    COMPS = ["c_cnt", "c_vol", "c_pos"]
    BETA0 = np.array([0.3, 0.0, 0.3])

    bar1m = datasources["bar1m"]
    # 向前多取若干天数据作为平滑/学习的预热（官方模板说明允许“多取若干天数据作为缓冲”）；
    # 输出只保留 [start_date, end_date] 内的截面（与股票池内连接时自动截断）。
    LOOKBACK_DAYS = 10
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")
    levels = range(1, 6)
    weights = {1: math.exp(-1.2), 2: math.exp(-0.9), 3: math.exp(-0.6), 4: math.exp(-0.3), 5: 1.0}

    raw_cols = ",\n            ".join(
        [f"CAST(COALESCE(bid_num_orders{i}, 0) AS DOUBLE) AS bn{i}" for i in levels]
        + [f"CAST(COALESCE(ask_num_orders{i}, 0) AS DOUBLE) AS an{i}" for i in levels]
        + [f"CAST(COALESCE(bid_volume{i}, 0) AS DOUBLE) AS bv{i}" for i in levels]
        + [f"CAST(COALESCE(ask_volume{i}, 0) AS DOUBLE) AS av{i}" for i in levels]
    )
    bid_far = " + ".join(f"bn{i} * {weights[i]:.10f}" for i in levels)
    ask_far = " + ".join(f"an{i} * {weights[i]:.10f}" for i in levels)
    vol_terms = " + ".join(f"(bv{i} - av{i}) / (bv{i} + av{i} + 1e-8)" for i in levels)

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,
            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol,
            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,
            {raw_cols}
        FROM {bar1m}
    ),
    cte_minute AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            vol,
            amt,
            ({bid_far}) AS fb,
            ({ask_far}) AS fa,
            ({vol_terms}) / 5.0 AS s_vol,
            CASE
                WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0
                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px
        FROM cte_raw
    ),
    cte_snapshot AS (
        SELECT
            instrument_id,
            trading_day,
            hms,
            vol,
            amt,
            px,
            s_vol,
            (fb - fa) / (fb + fa + 1e-8) AS s_cnt,
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
        FROM cte_minute
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            AVG(s_cnt) AS f_cnt,
            AVG(s_vol) AS f_vol,
            SUM(amt) AS sum_amt,
            SUM(vol) AS sum_vol,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px
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
        f.f_cnt, f.f_vol, f.sum_amt, f.sum_vol, f.last_px
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["f_cnt", "f_vol", "sum_amt", "sum_vol", "last_px"]:
        df[c] = df[c].astype("float64")

    ok = (df["sum_amt"] > 0) & (df["sum_vol"] > 0) & (df["last_px"] > 0)
    df["vwap"] = (df["sum_amt"] / df["sum_vol"].where(ok, 1.0)).where(ok, np.nan)
    df["f_pos"] = np.where(ok, np.log(df["vwap"].where(ok, 1.0) / df["last_px"].where(ok, 1.0)), np.nan)
    df = df.replace([np.inf, -np.inf], np.nan)

    # ---------- 同日跨窗平滑 + 已实现收益（只用于估计历史回归统计量） ----------
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["seg"] = df["date"].dt.hour * 10000 + df["date"].dt.minute * 100
    df["day"] = df["date"].dt.normalize()
    grp = df.groupby("instrument", observed=True, sort=False)

    prev_day = grp["day"].shift(1)
    for c in ["f_cnt", "f_vol"]:
        prev = grp[c].shift(1)
        use_prev = (prev_day == df["day"]) & prev.notna() & df[c].notna()
        df[c] = np.where(use_prev, (df[c] + PREV_WEIGHT * prev) / (1.0 + PREV_WEIGHT), df[c])

    next_seg = {93000: 100000, 100000: 103000, 103000: 110000, 110000: 113000,
                113000: 133000, 133000: 140000, 140000: 143000, 143000: 150000}
    nseg = grp["seg"].shift(-1)
    nday = grp["day"].shift(-1)
    nvwap = grp["vwap"].shift(-1)
    base = df["vwap"] if TARGET_MODE == "vwap" else df["last_px"]
    ok_fwd = (nday == df["day"]) & (nseg == df["seg"].map(next_seg)) & (base > 0) & (nvwap > 0)
    df["fwd"] = np.where(ok_fwd, nvwap / base.where(ok_fwd, 1.0) - 1.0, np.nan)
    df = df[df["seg"] != 150000]

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    in_range = df["date"] >= pd.to_datetime(start_date)
    pooled = pd.merge(df.loc[in_range, ["date", "instrument", "f_cnt", "f_vol", "f_pos", "fwd"]], stk_pool,
                      how="inner", on=["date", "instrument"])
    warmup = df.loc[~in_range, ["date", "instrument", "f_cnt", "f_vol", "f_pos", "fwd"]]
    res = pd.concat([warmup, pooled], ignore_index=True)
    res = res.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)

    # ---------- 截面居中排名 ----------
    by_date = res.groupby("date")
    ranks = by_date[["f_cnt", "f_vol", "f_pos"]].rank(pct=True) - 0.5
    res["c_cnt"] = ranks["f_cnt"]
    res["c_vol"] = ranks["f_vol"]
    res["c_pos"] = ranks["f_pos"]
    res["y"] = by_date["fwd"].rank(pct=True) - 0.5
    X = res[COMPS].fillna(0.0)

    # ---------- 每个截面的回归统计量 A = X'X/n, b = X'y/n ----------
    k = len(COMPS)
    has_y = res["y"].notna()
    stat_cols = []
    stats = pd.DataFrame({"date": res["date"]})
    for i in range(k):
        for j in range(i, k):
            name = f"a{i}{j}"
            stats[name] = (X.iloc[:, i] * X.iloc[:, j]).where(has_y)
            stat_cols.append(name)
        name = f"b{i}"
        stats[name] = (X.iloc[:, i] * res["y"]).where(has_y)
        stat_cols.append(name)
    stats["n"] = has_y.astype("float64")
    agg = stats.groupby("date")[stat_cols + ["n"]].sum()
    per = agg[stat_cols].div(agg["n"].where(agg["n"] >= MIN_STOCKS), axis=0)

    # 只用过去：整体后移一个截面，再指数加权。
    hist = per.shift(1)
    smooth = hist.ewm(halflife=HALFLIFE, ignore_na=True).mean().ffill()
    n_obs = hist[stat_cols[0]].notna().cumsum()

    D = np.eye(k) / 12.0
    betas = np.tile(BETA0, (len(smooth), 1))
    smooth_vals = smooth.to_numpy()
    col_idx = {c: idx for idx, c in enumerate(stat_cols)}
    for r in range(len(smooth)):
        n = n_obs.iloc[r]
        row = smooth_vals[r]
        if n <= 0 or np.isnan(row).any():
            continue
        A = np.zeros((k, k))
        b = np.zeros(k)
        for i in range(k):
            for j in range(i, k):
                A[i, j] = A[j, i] = row[col_idx[f"a{i}{j}"]]
            b[i] = row[col_idx[f"b{i}"]]
        tau = PRIOR_STRENGTH / float(n)
        try:
            betas[r] = np.linalg.solve(A + tau * D, b + tau * D @ BETA0)
        except np.linalg.LinAlgError:
            pass
    beta_df = pd.DataFrame(betas, index=smooth.index, columns=COMPS)

    w = beta_df.reindex(res["date"]).to_numpy()
    res["factor"] = (X.to_numpy() * w).sum(axis=1)
    res["factor"] = res["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    res = res[res["date"] >= pd.to_datetime(start_date)]
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
