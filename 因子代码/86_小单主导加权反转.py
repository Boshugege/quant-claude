def main(datasources, start_date, end_date):
    """
    小单主导加权反转（参考 Llorente, Michaely, Saar & Wang, RFS 2002 及券商“单笔成交金额”系列思路）。

    原文结论：风险分担/噪声交易造成的价格变动倾向于反转，信息交易造成的倾向于延续。
    以平均单笔成交额（成交额 / 成交笔数，首次使用 deal_number 字段）衡量交易者结构：
    单笔成交额相对平时偏小，说明散户/噪声交易主导，价格偏离更可能反转。
    本因子：相对单笔成交额 = 当日开盘至今平均单笔成交额 / 过去 10 个交易日同一时刻的均值；
      factor = 80 号反转信号截面居中排名 x (1.5 - 相对单笔成交额的截面百分位)。
    单笔越小（散户越主导），反转权重越高。与 80 号对比可检验“小单主导时反转更强”。
    需要向前多取 25 个自然日数据；历史不足时权重取中性 1。
    同批 80–87 共用同一段 SQL：逐窗口聚合成交额、成交量、成交笔数、窗口末中间价、前收盘价、
    当日开盘价（额外聚合 14:30–15:00 窗口，用于滚动 VWAP、波动率与同时段收益，不进入输出），
    在 pandas 中构造各分量（均只用过去数据）：
      r_roll : 滚动 240 分钟 VWAP 偏离反转 ln(最近 8 个窗口 VWAP / 当前价)（71 号，0.78304）
      sigma  : 个股最近 40 个窗口（约 5 个交易日）窗口间对数收益的标准差（73 号的做法）
      r_rv   : r_roll / sigma，滚动基准 + 波动率标准化
      r_day  : 日内反转 ln(昨收 / 当前价)（56 号）
      r_30   : 30 分钟反转 ln(上一窗口末价 / 当前价)，当日首个窗口以昨收为基准（57 号）

    稳健性：最终缺失补 0（中性），每个截面缺失率为 0；向前取不到更早数据时各分量自动退化。
    性能：纯 SQL 向量化，读取 8 个字段，单次 GROUP BY，不使用任何窗口函数、不使用 Python UDF；
    跨窗口/跨日运算在 pandas 中对已聚合的小表（每股每天 9 行）完成。

    评测时平台会注入 datasources、start_date 和 end_date。
    返回且仅返回 ['date', 'instrument', 'factor'] 三列。
    """
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    # 向前多取若干天数据用于跨日计算（官方模板说明允许“多取若干天数据作为缓冲”）；
    # 输出只保留 [start_date, end_date] 内的截面（与股票池内连接时自动截断）。
    LOOKBACK_DAYS = 25
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d %H:%M:%S")

    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(bid_price1, 0) AS DOUBLE) AS bp1,
            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,
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
            vol,
            amt,
            deals,
            pre_close,
            CASE WHEN hms > '092500' AND hms <= '093100' THEN px_open ELSE 0 END AS open_part,
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
            SUM(amt) AS sum_amt,
            SUM(vol) AS sum_vol,
            SUM(deals) AS sum_deals,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(open_part) AS open_part
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
        f.sum_amt, f.sum_vol, f.sum_deals, f.last_px, f.pre_close, f.open_part
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """

    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    for c in ["sum_amt", "sum_vol", "sum_deals", "last_px", "pre_close", "open_part"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)
    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
    df["day"] = df["date"].dt.normalize()
    df["seg"] = df["date"].dt.hour * 10000 + df["date"].dt.minute * 100

    def safe_log_ratio(num, den):
        ok = (num > 0) & (den > 0)
        return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)

    grp = df.groupby("instrument", observed=True, sort=False)
    grp_day = df.groupby(["instrument", "day"], observed=True, sort=False)
    same_day = grp["day"].shift(1) == df["day"]

    # 滚动 240 分钟 VWAP 反转（71 号），取不到时退回开盘以来累计 VWAP（67 号）。
    cum_amt = grp_day["sum_amt"].cumsum()
    cum_vol = grp_day["sum_vol"].cumsum()
    day_vwap = (cum_amt / cum_vol.where(cum_vol > 0)).where(cum_amt > 0)
    roll_amt = grp["sum_amt"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    roll_vol = grp["sum_vol"].transform(lambda s: s.rolling(8, min_periods=1).sum())
    roll_vwap = (roll_amt / roll_vol.where(roll_vol > 0)).where(roll_amt > 0)
    df["r_roll"] = safe_log_ratio(roll_vwap, df["last_px"]).fillna(safe_log_ratio(day_vwap, df["last_px"]))

    # 个股波动率（73 号）：最近 40 个窗口窗口间对数收益的标准差；不足时用截面中位数。
    win_ret = safe_log_ratio(df["last_px"], grp["last_px"].shift(1))
    sigma = win_ret.groupby(df["instrument"], observed=True).transform(lambda s: s.rolling(40, min_periods=8).std())
    sigma = sigma.where(sigma > 0)
    sigma = sigma.fillna(sigma.groupby(df["date"]).transform("median"))
    df["sigma"] = sigma

    df["r_rv"] = (df["r_roll"] / df["sigma"]).fillna(df["r_roll"])
    prev_px = grp["last_px"].shift(1).where(same_day, df["pre_close"])
    df["r_day"] = safe_log_ratio(df["pre_close"], df["last_px"])
    df["r_30"] = safe_log_ratio(prev_px, df["last_px"])

    # 过去 N 个交易日同一时刻的取值均值（按（股票，时刻）分组，先后移一天再滚动，只用过去数据）。
    def past_same_seg_mean(col, n_days, min_days):
        return df.groupby(["instrument", "seg"], observed=True, sort=False)[col].transform(
            lambda s: s.shift(1).rolling(n_days, min_periods=min_days).mean()
        )
    cum_amt_today = grp_day["sum_amt"].cumsum()
    cum_deals_today = grp_day["sum_deals"].cumsum()
    df["avg_trade"] = (cum_amt_today / cum_deals_today.where(cum_deals_today > 0)).where(cum_amt_today > 0)
    past_trade = past_same_seg_mean("avg_trade", 10, 3)
    df["w_relsize"] = df["avg_trade"] / past_trade.where(past_trade > 0)

    df = df.replace([np.inf, -np.inf], np.nan)

    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")
    stk_pool = dai.query(
        f"SELECT date, instrument FROM {instruments_table}",
        filters={"date": [start_date, end_date]},
        compression=True,
    ).df()
    keep = [c for c in df.columns if c.startswith("r_") or c.startswith("w_")]
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool, how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"])

    # 反转信号（80 号）截面居中排名，乘以条件权重的截面百分位（缺失时取 0.5，即不调整）。
    z = result.groupby("date")["r_rv"].rank(pct=True) - 0.5
    w = result.groupby("date")["w_relsize"].rank(pct=True).fillna(0.5)
    result["factor"] = z * (1.5 - w)

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
