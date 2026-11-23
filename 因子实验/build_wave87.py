# -*- coding: utf-8 -*-
"""第八十七批：暴露表信息化的两个新同伴轴 + 四个分钟级微观结构新轴（全部无低通、独立取证版）。

设计约束（任务说明 21:00 方向调整）：
- 头部调参冻结，本批 6 个名额全部给新信息源，一律用无低通版本取证。
- 561/562 直接复用 497 的 SQL 原文（评测机 SQL 缓存可整批只执行一次），只改 pandas 段。
- 563-566 共用同一条新的分钟级 SQL（带窗口函数），同样只执行一次。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

S497 = (CODE / "497_单独行业内条件反转.py").read_text(encoding="utf-8")

ANCHOR = '''    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
assert ANCHOR in S497
KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev"]'
assert KEEP_OLD in S497
SIG_OLD = '    result["signal"] = result["c_condrev"]'
assert SIG_OLD in S497
VER = '''    result["factor"] = result["signal"]
    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面。'''
assert VER in S497

# ---------- 561：行业内前一日市值加权同伴残差反转 ----------
BLOCK_561 = '''
    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]
    _res66 = []
    for _c66 in ["r_30", "r_rv4", "r_rv8", "r_day"]:
        _v66 = df[_c66]
        _ok66 = (_v66.notna() & (_mkw > 0)).to_numpy()
        _vk = np.where(_ok66, np.nan_to_num(_v66.to_numpy()), 0.0)
        _wk = np.where(_ok66, _mkw.to_numpy(), 0.0)
        _gsum = pd.Series(_vk * _wk, index=df.index).groupby(_pk66).transform("sum").to_numpy()
        _gw = pd.Series(_wk, index=df.index).groupby(_pk66).transform("sum").to_numpy()
        _self66 = _vk * _wk
        _den66 = _gw - _wk
        _peer66 = np.where(_den66 > 0, (_gsum - _self66) / np.where(_den66 > 0, _den66, 1.0), np.nan)
        _res66.append(pd.Series(_v66.to_numpy() - _peer66, index=df.index))
    _rkb = pd.concat([x.groupby(df["date"]).rank(pct=True) for x in _res66], axis=1).mean(axis=1) - 0.5
    df["c_indpeer"] = -_rkb
'''
# ---------- 562：行业内前一日市值加权同伴成交额偏离 ----------
BLOCK_562 = '''
    # ===== 新轴 562：行业内「前一日市值加权同伴」成交额偏离（负向：相对同伴成交萎缩 = 关注度低） =====
    _mkw = df["mcap"].fillna(0.0)
    _ind66 = df["ind_code"]
    _amt66 = df["sum_amt"].fillna(0.0)
    _ww = _mkw.where(_mkw > 0, 0.0)
    _num66 = (_amt66 * _ww).groupby([df["date"], _ind66]).transform("sum")
    _den66 = _ww.groupby([df["date"], _ind66]).transform("sum")
    _self66 = _amt66 * _ww
    _dw66 = (_den66 - _ww)
    _peer66 = (_num66 - _self66) / _dw66.where(_dw66 > 0)
    _dev66 = np.log((_amt66 + 1.0) / (_peer66.where(_peer66 > 0) + 1.0))
    df["c_amtsize"] = -_dev66.groupby(df["instrument"], sort=False).transform(
        lambda x: x.ewm(span=10, min_periods=8).mean())
'''

for num, block, col, fname, desc in [
    ("561", BLOCK_561, "c_indpeer", "561_单独行业内市值加权同伴残差反转.py",
     "行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）"),
    ("562", BLOCK_562, "c_amtsize", "562_单独行业内市值加权成交额偏离.py",
     "行业内前一日市值加权同伴成交额偏离（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）"),
]:
    body = S497.replace(ANCHOR, ANCHOR + "\n" + block)
    body = body.replace(KEEP_OLD, KEEP_OLD[:-2] + ', "%s"]' % col if False else '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "%s"]' % col)
    body = body.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)

# ---------- 563-566：分钟级微观结构共用 SQL ----------
MIN_SQL = '''    sql = f"""
    WITH cte_raw AS (
        SELECT
            instrument_id,
            strftime(date, '%Y-%m-%d') AS trading_day,
            strftime(date, '%H%M%S') AS hms,
            CAST(COALESCE(high, 0) AS DOUBLE) AS px_high,
            CAST(COALESCE(low, 0) AS DOUBLE) AS px_low,
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
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
            px_high,
            px_low,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN amt ELSE 0
            END AS amt_c,
            CASE WHEN px_close > 0 THEN px_close ELSE 0 END AS px,
            CASE WHEN px_high > 0 AND px_low > 0 AND px_high = px_low THEN 1 ELSE 0 END AS flat_c,
            CASE WHEN vol > 0 THEN 1 ELSE 0 END AS trad_c,
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
    cte_seq AS (
        SELECT
            instrument_id, trading_day, hms, pre_close, px_high, px_low, amt_c, px, flat_c, trad_c, time_segment,
            LAG(px) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS px_l1,
            LAG(px, 2) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS px_l2,
            LAG(amt_c) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS amt_l1
        FROM cte_snapshot
    ),
    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt_c) AS sum_amt,
            SUM(amt_c * amt_c) AS sum_amt2,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND (px - px_l1) <> 0 AND (amt_c - amt_l1) <> 0 AND (px - px_l1) * (amt_c - amt_l1) > 0 THEN 1 ELSE 0 END) AS n_vpsync,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND (px - px_l1) <> 0 AND (amt_c - amt_l1) <> 0 THEN 1 ELSE 0 END) AS n_vppair,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 AND (px - px_l1) <> 0 AND (px_l1 - px_l2) <> 0 AND (px - px_l1) * (px_l1 - px_l2) > 0 THEN 1 ELSE 0 END) AS n_ppsame,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 AND (px - px_l1) <> 0 AND (px_l1 - px_l2) <> 0 THEN 1 ELSE 0 END) AS n_pppair,
            SUM(flat_c) AS n_flat,
            SUM(trad_c) AS n_trad,
            COUNT(*) AS n_min,
            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close,
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low
        FROM cte_seq
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
        f.sum_amt, f.sum_amt2, f.n_vpsync, f.n_vppair, f.n_ppsame, f.n_pppair, f.n_flat, f.n_trad, f.n_min,
        f.last_px, f.pre_close, f.win_high, f.win_low
    FROM cte_window f
    LEFT JOIN all_instruments USING (instrument_id)
    """
'''

MICRO_HEAD = '''def main(datasources, start_date, end_date):
    """%s（09:30 置零版）。

    分钟级微观结构新轴，只用 1 分钟行情；不带任何头部叠加、不带低通。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]

    LOOKBACK_DAYS = 400
    query_start = (pd.to_datetime(start_date) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime("%%Y-%%m-%%d %%H:%%M:%%S")

'''

MICRO_TAIL = '''
    df = dai.query(sql, filters={"date": [query_start, end_date]}, compression=True).df()
    df = df.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    for c in ["sum_amt", "sum_amt2", "n_vpsync", "n_vppair", "n_ppsame", "n_pppair", "n_flat", "n_trad", "n_min",
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

    # 窗口内分钟结构指标
    _amt = df["sum_amt"].where(df["sum_amt"] > 0)
    _hhi = df["sum_amt2"] / (_amt * _amt)
    _sync = df["n_vpsync"] / df["n_vppair"].where(df["n_vppair"] > 0)
    _pp = df["n_ppsame"] / df["n_pppair"].where(df["n_pppair"] > 0)
    _flat = df["n_flat"] / df["n_min"].where(df["n_min"] > 0)
    _halt = 1.0 - df["n_trad"] / df["n_min"].where(df["n_min"] > 0)

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

MICRO = [
    ("563", "c_hhi", "563_单独分钟成交集中度.py", "窗口内分钟成交额集中度（HHI）的 40 窗口均值，取负：分散成交偏反转",
     '    df["c_hhi"] = _rk(_smooth(_hhi), -1.0)'),
    ("564", "c_vpsync", "564_单独量价同步性.py", "窗口内分钟成交额变动与价格变动同号的占比，40 窗口均值取负：量价共振偏趋势",
     '    df["c_vpsync"] = _rk(_smooth(_sync), -1.0)'),
    ("565", "c_ppsame", "565_单独分钟价格路径持续性.py", "窗口内相邻分钟收益同号占比，40 窗口均值取负：路径越持续越偏趋势",
     '    df["c_ppsame"] = _rk(_smooth(_pp), -1.0)'),
    ("566", "c_flat", "566_单独一字板停牌代理.py", "窗口内一字分钟（最高=最低）占比与零成交分钟占比的合成，40 窗口均值取负",
     '    df["c_flat"] = _rk(_smooth(_flat), -1.0) + 0.5 * _rk(_smooth(_halt), -1.0)'),
]

for num, col, fname, desc, sig in MICRO:
    body = MICRO_HEAD % desc + MIN_SQL + MICRO_TAIL % (sig, col, col)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
