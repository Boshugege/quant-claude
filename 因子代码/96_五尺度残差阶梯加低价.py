def main(datasources, start_date, end_date):
    """
    93 号阶梯 + 低价 合成（新方向：低价股效应）。

    构造：阶梯 = 93 号（五尺度原始阶梯 + 残差阶梯合成，不分时段）；factor = 截面排名加权平均：阶梯 1.0、低价 0.2。
    本地 IC（开发 / 验证 / 保留）：0.0597 / 0.0538 / 0.0430；2024 保留集 IC、IC_IR 在单一附加项候选中最高。
    依据：低价股价格离散度大、散户占比高，短期偏离回吐更明显；低价单独 IC 约 0.02–0.03，换手极低。
    同批 94–97 由本地评分系统筛选（见《本地评分系统说明.md》第 8.4 节）：在开发集 2020–2022、
    验证集 2023、保留集 2024 上均优于 93 号与 90 号（公榜 0.81328）。与 88–93 共用同一段 SQL，分量如下（均只用过去数据）：
      r_30   : 30 分钟反转 ln(上一窗口末价 / 当前价)，当日首个窗口以昨收为基准（57 号）
      r_vw   : 本窗口 VWAP 偏离 ln(窗口 VWAP / 当前价)（39 号）
      r_rv2/4/8 : 滚动 2/4/8 个窗口（1/2/4 小时）成交 VWAP 偏离 ln(滚动 VWAP / 当前价)，跨日滚动
      r_day  : 日内反转 ln(昨收 / 当前价)（56 号）
      sigma  : 个股最近 40 个窗口（约 5 个交易日）窗口间对数收益的标准差（73 号做法）
    数据清洗：成交均价 amount/volume 落在该分钟 [最低价 x 0.98, 最高价 x 1.02] 之外的分钟
    （本地数据中多见于 15:00，疑似盘后/大宗成交计入，VWAP 可偏离数倍）不计入任何 VWAP 累加。

    稳健性：最终缺失补 0（中性），每个截面缺失率为 0；向前取不到更早数据时各分量自动退化。
    性能：纯 SQL 向量化，读取 9 个字段，单次 GROUP BY，不使用任何窗口函数、不使用 Python UDF；
    跨窗口/跨日运算在 pandas 中对已聚合的小表（每股每天 9 行）完成。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取若干天数据用于滚动 VWAP 与波动率（官方模板说明允许“多取若干天数据作为缓冲”）；
    # 输出只保留 [start_date, end_date] 内的截面（与股票池内连接时自动截断）。
    LOOKBACK_DAYS = 10
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
    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    def vwap_of(amt, vol):
        return (amt / vol.where(vol > 0)).where(amt > 0)

    grp = df.groupby("instrument", observed=True, sort=False)
    grp_day = df.groupby(["instrument", "day"], observed=True, sort=False)
    same_day = grp["day"].shift(1) == df["day"]
    px = df["last_px"]

    day_vwap = vwap_of(grp_day["sum_amt"].cumsum(), grp_day["sum_vol"].cumsum())
    r_dvwap = safe_log_ratio(day_vwap, px)

    def rolling_vwap_rev(k):
        roll_amt = grp["sum_amt"].transform(lambda s: s.rolling(k, min_periods=1).sum())
        roll_vol = grp["sum_vol"].transform(lambda s: s.rolling(k, min_periods=1).sum())
        return safe_log_ratio(vwap_of(roll_amt, roll_vol), px).fillna(r_dvwap)

    win_ret = safe_log_ratio(px, grp["last_px"].shift(1))
    sigma = win_ret.groupby(df["instrument"], observed=True).transform(lambda s: s.rolling(40, min_periods=8).std())
    sigma = sigma.where(sigma > 0)
    sigma = sigma.fillna(sigma.groupby(df["date"]).transform("median"))

    comps = {
        "r_30": safe_log_ratio(grp["last_px"].shift(1).where(same_day, df["pre_close"]), px),
        "r_vw": safe_log_ratio(vwap_of(df["sum_amt"], df["sum_vol"]), px),
        "r_rv2": rolling_vwap_rev(2),
        "r_rv4": rolling_vwap_rev(4),
        "r_rv8": rolling_vwap_rev(8),
        "r_day": safe_log_ratio(df["pre_close"], px),
    }
    for name, s in comps.items():
        df[name] = s
        df[name + "_s"] = (s / sigma).fillna(s)

    # 残差化（93 号使用）：大盘收益 = 每个截面全部股票窗口收益的等权均值；个股 beta = 最近 40 个窗口
    # 个股收益对大盘收益的滚动回归系数（至少 10 个窗口，截断到 [-1, 4]，不足时取 1）。
    # 分量 s 的残差 = s - beta x 截面均值(s)：去掉“跟随大盘”的部分，只保留个股自身的偏离。
    ret30 = safe_log_ratio(px, grp["last_px"].shift(1).where(same_day)).fillna(safe_log_ratio(px, df["pre_close"]))
    mkt30 = ret30.groupby(df["date"]).transform("mean")

    def roll_mean(s):
        return s.groupby(df["instrument"], observed=True).transform(lambda x: x.rolling(40, min_periods=10).mean())

    ex, ey, exy, ey2 = roll_mean(ret30), roll_mean(mkt30), roll_mean(ret30 * mkt30), roll_mean(mkt30 ** 2)
    var_m = (ey2 - ey ** 2).where((ey2 - ey ** 2) > 1e-12)
    beta = ((exy - ex * ey) / var_m).clip(-1, 4).fillna(1.0)
    for name in ["r_30", "r_rv2", "r_rv4", "r_rv8", "r_day"]:
        df[name + "_res"] = df[name] - beta * df[name].groupby(df["date"]).transform("mean")

    # 股票特征分量（94–96 使用）：
    #   非流动性（Amihud）：最近 40 个窗口“|窗口收益| / 窗口成交额（亿元）”的均值（至少 10 个窗口）；越不流动越大。
    #   低价：-ln(当前价)；价格越低越大。
    amihud_raw = win_ret.abs() / (df["sum_amt"] / 1e8).where(df["sum_amt"] > 0)
    df["r_illiq"] = amihud_raw.groupby(df["instrument"], observed=True).transform(
        lambda x: x.rolling(40, min_periods=10).mean())
    df["r_lowpx"] = -np.log(px.where(px > 0))
    df = df.replace([np.inf, -np.inf], np.nan)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    keep = [c for c in df.columns if c.startswith("r_")]
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"]).reset_index(drop=True)

    def rank_blend(cols, weights=None):
        """各分量在股票池内逐截面居中排名后加权平均（缺失分量不参与，权重按可用分量重新归一）。"""
        weights = weights or [1.0] * len(cols)
        ranks = (result.groupby("date")[cols].rank(pct=True) - 0.5).to_numpy()
        w = np.array(weights, dtype=float)
        has = ~np.isnan(ranks)
        den = (has * w).sum(axis=1)
        num = np.nansum(np.nan_to_num(ranks) * w, axis=1)
        return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=result.index)

    def ladder93(parts):
        """93 号结构：原始阶梯与残差阶梯各自截面排名等权，再合成。"""
        result["_a"] = rank_blend(["r_" + n for n in parts])
        result["_b"] = rank_blend(["r_" + n + "_res" for n in parts])
        return rank_blend(["_a", "_b"])

    def ladder_by_slot():
        """按时段调整：14:00 截面只用短周期分量（30 分钟、1 小时、2 小时），其余时段用五尺度。"""
        full = ladder93(["30", "rv2", "rv4", "rv8", "day"])
        short = ladder93(["30", "rv2", "rv4"])
        is_1400 = (result["date"].dt.hour == 14) & (result["date"].dt.minute == 0)
        return full.where(~is_1400, short)

    result["ladder"] = ladder93(["30", "rv2", "rv4", "rv8", "day"])
    result["factor"] = rank_blend(["ladder", "r_lowpx"], [1.0, 0.2])

    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    return result[["date", "instrument", "factor"]].sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)


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
