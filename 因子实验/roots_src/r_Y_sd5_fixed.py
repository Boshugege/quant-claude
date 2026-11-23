# -*- coding: utf-8 -*-
"""【近 5 日单笔成交额离散度 Y_sd5（平台可算版）】

原定义（1320_多日分钟框架 STAT="Y_sd5"）：
    STDDEV_SAMP(LN(单笔成交额))  窗口 = PARTITION BY instrument_id ORDER BY date ROWS BETWEEN 1199 PRECEDING AND CURRENT ROW
    —— 平台上跨日 1200 分钟窗口返回全空，这条根在合成里贡献为零。

本文件把 1200 行窗口拆成"按日聚合 + pandas 跨日滚动"，**逐值等价**（99.5% 的 (股票, 交易日) 恰好 240 根 1 分钟 bar，
240×5 = 1200）：样本标准差用 (n, Σx, Σx²) 合并即可跨日相加：
    窗口(D, 截面 s) = 当日至今(D, s) + Σ_{j=1..4} 全日(D-j) + [全日(D-5) − 当日至今(D-5, s)]

与 1320 框架相同口径：不含集合竞价 bar、只用 1 分钟 bar、不读 vwap／暴露表。
"""
import numpy as np
import pandas as pd

SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]


def day_agg_sql(bar1m):
    """按 (股票, 交易日) 聚合：全日 + 各截面的"当日至今"，只用到日内窗口（平台可算）。"""
    return f"""
    WITH b AS (
        SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d, strftime(b.date, '%H%M') AS hm,
            CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c,
            CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a, CAST(COALESCE(b.deal_number, 0) AS DOUBLE) AS n
        FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '150000'
          AND b.close > 0 AND b.open > 0
    ), m1 AS (
        SELECT *, LN(c / COALESCE(LAG(c) OVER (PARTITION BY instrument_id, d ORDER BY date), o)) AS r,
            CASE WHEN n > 0 AND a > 0 THEN LN(a / n) END AS lats
        FROM b
    ), m2 AS (
        SELECT *,
            AVG(lats) OVER (PARTITION BY instrument_id, d ORDER BY date
                            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS lats_prev
        FROM m1
    ), m3 AS (
        SELECT instrument_id, instrument, d, date, hm,
            SUM(r * (lats - lats_prev)) OVER w AS cnd,
            SUM(ABS(r)) OVER w AS cabs,
            COUNT(lats) OVER w AS cn,
            SUM(lats) OVER w AS sl,
            SUM(lats * lats) OVER w AS sl2
        FROM m2
        WINDOW w AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    )
    SELECT * FROM m3 WHERE hm IN ('1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500')
    """


def roll5(df, col):
    """把"当日至今"列按 (股票, 交易日) 展开，返回 {截面: DataFrame(d × instrument)} 的 5 日窗口和。"""
    w = df.set_index(["instrument", "d", "hm"])[col].unstack("hm").sort_index()
    full = w["1500"]
    prev4 = full.shift(1) + full.shift(2) + full.shift(3) + full.shift(4)
    out = {}
    for s in SLOTS:
        out[s] = w[s] + prev4 + full.shift(5) - w[s].shift(5)
    return out


def main(datasources, start_date, end_date):
    import dai

    bar1m = datasources["bar1m"]
    q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=40)).strftime("%Y-%m-%d %H:%M:%S")
    df = dai.query(day_agg_sql(bar1m), filters={"date": [q0, end_date]}, compression=True).df()
    if not len(df):
        return pd.DataFrame(columns=["date", "instrument", "factor"])
    df["instrument"] = df["instrument"].astype(str)
    df["hm"] = df["hm"].astype(str)
    df["d"] = pd.to_datetime(df["d"])
    cn, sl, sl2 = roll5(df, "cn"), roll5(df, "sl"), roll5(df, "sl2")
    out = []
    for s in SLOTS:
        n = cn[s].where(cn[s] > 1)
        sig = np.sqrt((sl2[s] - sl[s] ** 2 / n) / (n - 1))
        t = sig.rename("signal").reset_index()
        t = t[t["d"] >= pd.Timestamp(pd.to_datetime(start_date).normalize())]
        t["date"] = t["d"] + pd.Timedelta(hours=int(s[:2]), minutes=int(s[2:]))
        out.append(t[["date", "instrument", "signal"]])
    df = pd.concat(out, ignore_index=True)
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
    from bigmodule import M
    import structlog

    logger = structlog.get_logger()
    datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
    start_date = "2023-01-01 00:00:00"
    end_date = "2023-03-01 23:59:59"
    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
