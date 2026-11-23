# -*- coding: utf-8 -*-
"""【第十八轮 N3：资金流族的两个新方向（只用日内 SQL 窗口，不用跨日分钟窗口）】

背景（第八轮／第六轮记录）：资金流族的"窗口长度"自由度已被走完（30/60/90/120、按成交额、跨日 bar 全是同一件事），
`Z_mfr60`（近 60 分钟上涨分钟成交量占比，反用）在公榜上**完全兑现**（1362/1363 各 +0.010/+0.003）。
用户要求 N3 提 1–2 个**不依赖季节性、不用跨日分钟窗口**的新方向，参考资金流族扩展。

本文件给三个候选（同一段日内 SQL，变体只改 STAT 一行）：

  Z_mfr60c  近 60 分钟上涨量占比——**对照组**，与 1334 的 `Z_mfr60` 逐值相同（用来验证本文件口径无误）
  Z_bigup   **按成交笔数分层**的资金流：分钟单笔额（amount ÷ deal_number）在当日至今均值之上的算"大单分钟"，
            之下的算"小单分钟"，取近 60 分钟两者的上涨量占比之差（大单流 − 小单流）
  Z_bigsh   **大单参与度**：近 60 分钟"大单分钟"成交量占全部成交量的比例（与涨跌方向无关）
  Z_flowgap **日内累积形态**：近 60 分钟上涨量占比 − 当日至今上涨量占比（资金流的"加速/减速"）
  Z_mfr5d   **近 5 日上涨量占比**：SQL 按（股票, 交易日）聚合，pandas 跨日滚动 5 日——验证 N1 的"按日聚合＋滚动"机制可复用

口径与 1334／1273 相同：1 分钟 bar 用结束时刻标注，不含集合竞价；不读 vwap／暴露表；不写自身收益反转。
"""
import numpy as np
import pandas as pd

SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
STAT = "Z_bigup"


def day_agg_sql(bar1m):
    return f"""
    WITH b AS (
        SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d, strftime(b.date, '%H%M') AS hm,
            CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c,
            CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
            CAST(COALESCE(b.deal_number, 0) AS DOUBLE) AS n
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
        SELECT instrument_id, instrument, d, hm, date,
            SUM(v) OVER w AS vall,
            SUM(CASE WHEN r > 0 THEN v ELSE 0 END) OVER w AS vup,
            SUM(v) OVER w60 AS vall60,
            SUM(CASE WHEN r > 0 THEN v ELSE 0 END) OVER w60 AS vup60,
            SUM(CASE WHEN lats >= lats_prev THEN v ELSE 0 END) OVER w60 AS vbig60,
            SUM(CASE WHEN r > 0 AND lats >= lats_prev THEN v ELSE 0 END) OVER w60 AS vbigup60,
            SUM(CASE WHEN lats < lats_prev THEN v ELSE 0 END) OVER w60 AS vsml60,
            SUM(CASE WHEN r > 0 AND lats < lats_prev THEN v ELSE 0 END) OVER w60 AS vsmlup60,
            SUM(n) OVER w AS nall, SUM(CASE WHEN r > 0 THEN n ELSE 0 END) OVER w AS nup,
            SUM(n) OVER w60 AS nall60, SUM(CASE WHEN r > 0 THEN n ELSE 0 END) OVER w60 AS nup60,
            SUM(CASE WHEN r > 0 THEN v WHEN r < 0 THEN -v ELSE 0 END) OVER w AS vnet
        FROM m2
        WINDOW w AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
               w60 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
    )
    SELECT * FROM m3 WHERE hm IN ('1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500')
    """


def roll5(df, col):
    w = df.set_index(["instrument", "d", "hm"])[col].unstack("hm").sort_index()
    full = w["1500"]
    prev4 = full.shift(1) + full.shift(2) + full.shift(3) + full.shift(4)
    return {s: w[s] + prev4 + full.shift(5) - w[s].shift(5) for s in SLOTS}


def signals(df):
    with np.errstate(all="ignore"):
        mfr60 = df["vup60"] / df["vall60"].where(df["vall60"] > 0)
        return {
            "Z_mfr60c": mfr60,
            "Z_bigsh": df["vbig60"] / df["vall60"].where(df["vall60"] > 0),
            "Z_bigup": (df["vbigup60"] / df["vbig60"].where(df["vbig60"] > 0)
                        - df["vsmlup60"] / df["vsml60"].where(df["vsml60"] > 0)),
            "Z_flowgap": mfr60 - df["vup"] / df["vall"].where(df["vall"] > 0),
            "Z_cmfr60": df["nup60"] / df["nall60"].where(df["nall60"] > 0),
            "Z_vnet": df["vnet"] / df["vall"].where(df["vall"] > 0),
        }


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

    out = []
    if STAT in ("Z_d1mfr", "Z_d1net"):
        # 前一交易日的"全天"值，当日所有截面同值（日内不变锚，用 K2 的降换手机制）
        w = df.set_index(["instrument", "d", "hm"])
        alld = w["vall"].unstack("hm").sort_index()
        numd = w["vup" if STAT == "Z_d1mfr" else "vnet"].unstack("hm").sort_index()
        base = numd["1500"] / alld["1500"].where(alld["1500"] > 0)
        prev = base.groupby(level=0).shift(1)
        for s in SLOTS:
            t = prev.rename("signal").reset_index()
            t = t[t["d"] >= pd.Timestamp(pd.to_datetime(start_date).normalize())]
            t["date"] = t["d"] + pd.Timedelta(hours=int(s[:2]), minutes=int(s[2:]))
            out.append(t[["date", "instrument", "signal"]])
    elif STAT in ("Z_mfr5d", "Z_net5d"):
        up, al = roll5(df, "vup"), roll5(df, "vall")
        net = roll5(df, "vnet")
        for s in SLOTS:
            sig = (up[s] / al[s].where(al[s] > 0)) if STAT == "Z_mfr5d" else (net[s] / al[s].where(al[s] > 0))
            t = sig.rename("signal").reset_index()
            t = t[t["d"] >= pd.Timestamp(pd.to_datetime(start_date).normalize())]
            t["date"] = t["d"] + pd.Timedelta(hours=int(s[:2]), minutes=int(s[2:]))
            out.append(t[["date", "instrument", "signal"]])
    else:
        sig = signals(df)[STAT]
        t = sig.rename("signal").to_frame().join(df[["instrument", "d", "hm"]]).reset_index(drop=True)
        t = t[t["hm"].isin(SLOTS)]
        t = t[t["d"] >= pd.Timestamp(pd.to_datetime(start_date).normalize())]
        t["date"] = t["d"] + pd.to_timedelta(t["hm"].str[:2].astype(int), unit="h") + pd.to_timedelta(t["hm"].str[2:].astype(int), unit="m")
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
