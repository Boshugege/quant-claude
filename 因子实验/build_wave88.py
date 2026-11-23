# -*- coding: utf-8 -*-
"""第八十八批：五条新信息源独立取证（全部无低通）。
568/569/570 = 同伴定义再换三个（复用 497 的 SQL 文本，评测机命中缓存）；
571/572 = 涨停打开比例 / 分钟成交笔数集中度（共用一条新的分钟级 SQL）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
B561 = '''    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
assert B561 in S561

VAR = [
    ("568", "568_单独全市场市值加权同伴残差反转.py",
     '''    # ===== 新轴 568：全市场「前一日市值加权同伴」残差反转（同伴定义 = 全市场，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values]''',
     "全市场前一日市值加权同伴残差反转（不按行业切分，权重 = 前一日流通市值；暴露表全部按交易日错一天）"),
    ("569", "569_单独行业内小市值加权同伴残差反转.py",
     '''    # ===== 新轴 569：行业内「小市值加权同伴」残差反转（权重 = 1 / 前一日流通市值） =====
    _mkw = (1.0 / df["mcap"].where(df["mcap"] > 0)).fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]''',
     "行业内小市值加权同伴残差反转（同伴定义 = 申万一级行业内，权重 = 1/前一日流通市值；暴露表按交易日错一天）"),
    ("570", "570_单独行业内高换手加权同伴残差反转.py",
     '''    # ===== 新轴 570：行业内「前一日换手加权同伴」残差反转（权重 = 当日成交额 / 前一日流通市值） =====
    _dly70 = df.groupby(["day", "instrument"], sort=False).agg(
        _mcap70=("mcap", "first"), _amt70=("sum_amt", "sum")).reset_index()
    _dly70["_turn70"] = _dly70["_amt70"] / _dly70["_mcap70"].where(_dly70["_mcap70"] > 0)
    _dly70 = _dly70.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dly70["_turn70"] = _dly70.groupby("instrument", sort=False)["_turn70"].shift(1)
    _k70 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["_turnlag"] = pd.Series(_dly70.set_index(["instrument", "day"])["_turn70"].reindex(_k70).to_numpy(), index=df.index)
    _mkw = df["_turnlag"].where(df["_turnlag"] > 0).fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]''',
     "行业内前一日换手加权同伴残差反转（权重 = 前一日真实换手率 = 成交额/前一日流通市值；暴露表按交易日错一天）"),
]
for num, fname, block, desc in VAR:
    body = S561.replace(B561, block)
    body = body.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", desc, 1)
    # 561 的列名/信号保持（c_indpeer、符号为 561 的方向），本批先按 561 的负号取证，若 IC 显著为负再翻转
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)

# ---------- 571/572：一条新的分钟级 SQL（含笔数、涨停打开） ----------
MIN2 = '''    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
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
            CASE WHEN px_close > 0 THEN px_close ELSE 0 END AS px,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high = px_low THEN 1 ELSE 0 END AS flat_c,
            CASE WHEN pre_close > 0 AND px_close >= pre_close * 1.095 THEN 1 ELSE 0 END AS lim_c,
            CASE WHEN pre_close > 0 AND px_close >= pre_close * 1.095 AND px_high > px_close THEN 1 ELSE 0 END AS limopen_c,
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
            SUM(amt_c * amt_c) AS sum_amt2,
            SUM(deals) AS sum_deals,
            SUM(deals * deals) AS sum_deals2,
            SUM(flat_c) AS n_flat,
            SUM(lim_c) AS n_lim,
            SUM(limopen_c) AS n_limopen,
            COUNT(*) AS n_min,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low
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
        f.sum_amt, f.sum_amt2, f.sum_deals, f.sum_deals2, f.n_flat, f.n_lim, f.n_limopen, f.n_min,
        f.last_px, f.pre_close, f.win_high, f.win_low
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """
'''
HEAD2 = '''def main(datasources, start_date, end_date):
    """%s（09:30 置零版）。

    分钟级新轴，只用 1 分钟行情；不带头部叠加、不带低通。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    LOOKBACK_DAYS = 400
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%%Y-%%m-%%d %%H:%%M:%%S")

'''
TAIL2 = '''
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_amt2", "sum_deals", "sum_deals2", "n_flat", "n_lim", "n_limopen", "n_min",
              "last_px", "pre_close", "win_high", "win_low"]:
        df[c] = df[c].astype("float64")
    df["last_px"] = df["last_px"].where(df["last_px"] > 0)
    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)

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
    df = df.replace([np.inf, -np.inf], np.nan)

    _deals = df["sum_deals"].where(df["sum_deals"] > 0)
    _dealshhi = df["sum_deals2"] / (_deals * _deals)
    _limopen = df["n_limopen"] / df["n_lim"].where(df["n_lim"] > 0)

    def _smooth(s):
        return s.groupby(df["instrument"], sort=False).transform(
            lambda x: x.rolling(40, min_periods=10).mean())

    def _rk(col, sign):
        return sign * (col.groupby(df["date"]).rank(pct=True) - 0.5)

%s
    keep = ["%s"]
    result = pd.merge(df[["date", "instrument"] + keep], stk_pool[["date", "instrument"]], how="inner", on=["date", "instrument"])
    result = result.drop_duplicates(["date", "instrument"]).sort_values(["date", "instrument"]).reset_index(drop=True)
    result["signal"] = result["%s"]
    result["factor"] = result["signal"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面。
    result.loc[result["date"].dt.strftime("%%H%%M") == "0930", "factor"] = 0.0
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
'''
for num, col, fname, desc, sig in [
    ("571", "c_limopen", "571_单独涨停打开比例.py", "窗口内触及涨停但打开（最高>收盘）的分钟占比，40 窗口均值取负",
     '    df["c_limopen"] = _rk(_smooth(_limopen), -1.0)'),
    ("572", "c_dealshhi", "572_单独分钟成交笔数集中度.py", "窗口内分钟成交笔数集中度（HHI），40 窗口均值取负",
     '    df["c_dealshhi"] = _rk(_smooth(_dealshhi), -1.0)'),
]:
    (CODE / fname).write_text(HEAD2 % desc + MIN2 + TAIL2 % (sig, col, col), encoding="utf-8")
    print("写出", fname)
