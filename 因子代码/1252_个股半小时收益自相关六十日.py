def main(datasources, start_date, end_date):
    """【新根因子 第三轮：多统计量框架】一份代码覆盖 L/M/N/O/P/Q/R/S/T/U/V 十一个方向，STAT 选一个统计量。
    同一段 SQL 被所有变体共用（评测机按 SQL 文本缓存），变体只改 STAT 一行。

    分钟口径：1 分钟 bar 用结束时刻标注，截面 T 取标签恰为 T 的那条 bar（含）之前的连续竞价分钟，
    不含集合竞价 bar（09:30 及之前）；"当日至今"的量都按 (股票, 交易日) 分区、按时间累计，不跨日。
    30 分钟段口径：(09:30,10:00] … (14:30,15:00]，只用截面所在交易日之前的完整交易日（多日统计）
    或截面时刻及之前的段（T_acc）。不读 vwap 表、不读暴露表、不含自身收益反转写法。

    STAT：
      L_lab10/20/40  下一个半小时段在过去 N 日的"段收盘 ÷ 段 VWAP − 1"均值（同时段季节性）
      L_ret20        下一个半小时段在过去 20 日的段收益均值；L_lab20_dm 去掉各时段平均
      M_q / M_q_lin  聪明钱：以 |r|/√v（或 |r|/v）加权的当日 VWAP ÷ 当日 VWAP − 1
      N_dir / N_big  大单方向：Σ r·(ln 单笔额 − 之前分钟均值) ÷ Σ|r|；大单分钟收益和 − 小单分钟收益和
      Q_eff / Q_seff / Q_ac1  路径效率 |净变动|÷Σ|r|、带符号版、分钟收益一阶自相关
      U_nj / U_sj    当日跳跃次数（|r| > 4 倍之前分钟波动）与带符号跳跃收益和
      S_osz / S_osz30  买方每笔委托挂单量 ÷ 卖方（1–3 档，对数）当日均值 / 近 30 分钟均值
      O_q5           一档买量 ÷ 卖量（对数）近 5 分钟均值
      T_acc          本段成交额 ÷ 上一段（对数）× 本段收益符号
      R_round / R_half  当前价相对最近整数元 / 半元的带符号距离（÷ 价格）
      V_spr          当日相对价差均值
      P_ac20 / P_ac60  个股过去 20/60 日相邻半小时段收益的一阶自相关（日内常数，供条件化头部用）
    """
    import numpy as np
    import pandas as pd
    import dai

    STAT = "P_ac60"
    SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
    NEXT = {"1000": "1030", "1030": "1100", "1100": "1130", "1130": "1330", "1330": "1400", "1400": "1430", "1430": "1500"}
    bar1m = datasources["bar1m"]
    fam = STAT.split("_")[0]

    if fam in ("M", "N", "Q", "U", "S", "O", "R", "V"):
        sql_min = f"""
        WITH b AS (
            SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d, strftime(b.date, '%H%M') AS hm,
                CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c,
                CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
                CAST(COALESCE(b.deal_number, 0) AS DOUBLE) AS n,
                CAST(COALESCE(b.bid_volume1, 0) + COALESCE(b.bid_volume2, 0) + COALESCE(b.bid_volume3, 0) AS DOUBLE) AS bv,
                CAST(COALESCE(b.ask_volume1, 0) + COALESCE(b.ask_volume2, 0) + COALESCE(b.ask_volume3, 0) AS DOUBLE) AS av,
                CAST(COALESCE(b.bid_num_orders1, 0) + COALESCE(b.bid_num_orders2, 0) + COALESCE(b.bid_num_orders3, 0) AS DOUBLE) AS bo,
                CAST(COALESCE(b.ask_num_orders1, 0) + COALESCE(b.ask_num_orders2, 0) + COALESCE(b.ask_num_orders3, 0) AS DOUBLE) AS ao,
                CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1, CAST(COALESCE(b.ask_volume1, 0) AS DOUBLE) AS av1,
                CAST(b.bid_price1 AS DOUBLE) AS bp1, CAST(b.ask_price1 AS DOUBLE) AS ap1
            FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
            WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '150000'
              AND b.close > 0 AND b.open > 0
        ), m1 AS (
            SELECT *, LN(c / COALESCE(LAG(c) OVER (PARTITION BY instrument_id, d ORDER BY date), o)) AS r,
                CASE WHEN n > 0 AND a > 0 THEN LN(a / n) END AS lats
            FROM b
        ), m2 AS (
            SELECT *,
                ABS(r) / SQRT(GREATEST(v, 1)) AS s1, ABS(r) / GREATEST(v, 1) AS s2,
                CASE WHEN bo > 0 AND ao > 0 AND bv > 0 AND av > 0 THEN LN((bv / bo) / (av / ao)) END AS osz,
                LN((bv1 + 1) / (av1 + 1)) AS q1,
                CASE WHEN ap1 > 0 AND bp1 > 0 AND ap1 >= bp1 THEN (ap1 - bp1) / ((ap1 + bp1) / 2) END AS spr,
                r * LAG(r) OVER (PARTITION BY instrument_id, d ORDER BY date) AS rr1,
                AVG(r * r) OVER (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS rv_prev,
                COUNT(r) OVER (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS n_prev,
                AVG(lats) OVER (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS lats_prev
            FROM m1
        ), m3 AS (
            SELECT instrument_id, instrument, d, hm, date, c,
                SUM(a) OVER w AS ca, SUM(v) OVER w AS cv,
                SUM(s1 * a) OVER w AS cs1a, SUM(s1 * v) OVER w AS cs1v, SUM(s2 * a) OVER w AS cs2a, SUM(s2 * v) OVER w AS cs2v,
                SUM(r * (lats - lats_prev)) OVER w AS cnd,
                SUM(CASE WHEN lats > lats_prev THEN r ELSE 0 END) OVER w AS cbig,
                SUM(CASE WHEN lats <= lats_prev THEN r ELSE 0 END) OVER w AS csmall,
                SUM(ABS(r)) OVER w AS cabs, SUM(r) OVER w AS csum, SUM(r * r) OVER w AS crr, SUM(rr1) OVER w AS crr1,
                SUM(CASE WHEN n_prev >= 10 AND ABS(r) > 4 * SQRT(rv_prev) THEN 1 ELSE 0 END) OVER w AS cnj,
                SUM(CASE WHEN n_prev >= 10 AND ABS(r) > 4 * SQRT(rv_prev) THEN r ELSE 0 END) OVER w AS csj,
                AVG(osz) OVER w AS mosz,
                AVG(osz) OVER (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) AS mosz30,
                AVG(q1) OVER (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS mq5,
                AVG(spr) OVER w AS mspr
            FROM m2
            WINDOW w AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        )
        SELECT * FROM m3 WHERE hm IN ('1000', '1030', '1100', '1130', '1330', '1400', '1430')
        """
        df = dai.query(sql_min, filters={"date": [start_date, end_date]}, compression=True).df()
        if not len(df):
            return pd.DataFrame(columns=["date", "instrument", "factor"])
        with np.errstate(all="ignore"):
            vw = df["ca"] / df["cv"].where(df["cv"] > 0)
            cabs = df["cabs"].where(df["cabs"] > 0)
            sig = {
                "M_q": (df["cs1a"] / df["cs1v"].where(df["cs1v"] > 0)) / vw - 1,
                "M_q_lin": (df["cs2a"] / df["cs2v"].where(df["cs2v"] > 0)) / vw - 1,
                "N_dir": df["cnd"] / cabs,
                "N_big": df["cbig"] - df["csmall"],
                "Q_eff": df["csum"].abs() / cabs,
                "Q_seff": df["csum"] / cabs,
                "Q_ac1": df["crr1"] / df["crr"].where(df["crr"] > 0),
                "U_nj": df["cnj"].astype(float),
                "U_sj": df["csj"],
                "S_osz": df["mosz"],
                "S_osz30": df["mosz30"],
                "O_q5": df["mq5"],
                "R_round": (df["c"] - np.round(df["c"])) / df["c"],
                "R_half": (df["c"] - np.round(df["c"] * 2) / 2) / df["c"],
                "V_spr": df["mspr"],
            }[STAT]
        df["signal"] = np.asarray(sig, dtype=float)
        df = df[["date", "instrument", "signal"]]
    else:
        q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=130)).strftime("%Y-%m-%d %H:%M:%S")
        sql_seg = f"""
        WITH b AS (
            SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d,
                CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c,
                CAST(COALESCE(b.volume, 0) AS DOUBLE) AS v, CAST(COALESCE(b.amount, 0) AS DOUBLE) AS a,
                CASE
                    WHEN strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '100000' THEN '1000'
                    WHEN strftime(b.date, '%H%M%S') > '100000' AND strftime(b.date, '%H%M%S') <= '103000' THEN '1030'
                    WHEN strftime(b.date, '%H%M%S') > '103000' AND strftime(b.date, '%H%M%S') <= '110000' THEN '1100'
                    WHEN strftime(b.date, '%H%M%S') > '110000' AND strftime(b.date, '%H%M%S') <= '113000' THEN '1130'
                    WHEN strftime(b.date, '%H%M%S') > '130000' AND strftime(b.date, '%H%M%S') <= '133000' THEN '1330'
                    WHEN strftime(b.date, '%H%M%S') > '133000' AND strftime(b.date, '%H%M%S') <= '140000' THEN '1400'
                    WHEN strftime(b.date, '%H%M%S') > '140000' AND strftime(b.date, '%H%M%S') <= '143000' THEN '1430'
                    WHEN strftime(b.date, '%H%M%S') > '143000' AND strftime(b.date, '%H%M%S') <= '150000' THEN '1500'
                END AS seg
            FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
            WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '150000'
              AND b.close > 0 AND b.open > 0
        )
        SELECT MAX(date) AS date, instrument_id, MAX(instrument) AS instrument, d, seg,
            SUM(a) AS amt, SUM(v) AS vol, arg_max(c, date) AS c_last, arg_min(o, date) AS o_first
        FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
        """
        sg = dai.query(sql_seg, filters={"date": [q0, end_date]}, compression=True).df()
        if not len(sg):
            return pd.DataFrame(columns=["date", "instrument", "factor"])
        sg["instrument"] = sg["instrument"].astype(str)
        sg["d"] = pd.to_datetime(sg["d"])
        sg = sg.sort_values(["instrument", "d", "seg"]).reset_index(drop=True)
        with np.errstate(all="ignore"):
            sg["lab"] = sg["c_last"] / (sg["amt"] / sg["vol"].where(sg["vol"] > 0)) - 1
            prev_c = sg.groupby(["instrument", "d"])["c_last"].shift(1)
            sg["ret"] = sg["c_last"] / prev_c.fillna(sg["o_first"]) - 1
        days = np.sort(sg["d"].unique())
        out_days = days[days >= pd.Timestamp(pd.to_datetime(start_date).normalize())]
        rows = []
        if fam == "L":
            col = "ret" if STAT.startswith("L_ret") else "lab"
            N = int("".join(ch for ch in STAT.split("_")[1] if ch.isdigit()))
            piv = sg.pivot_table(index="d", columns=["instrument", "seg"], values=col, aggfunc="last").reindex(days)
            mu = piv.rolling(N, min_periods=max(N // 2, 3)).mean().shift(1)       # 只用之前的交易日
            if STAT.endswith("_dm"):
                mu = mu - mu.T.groupby(level=0).transform("mean").T
            for t in SLOTS:
                nx = NEXT[t]
                s = mu.loc[out_days].xs(nx, axis=1, level=1)
                s = s.stack().rename("signal").reset_index()
                s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
                rows.append(s[["date", "instrument", "signal"]])
        elif fam == "P":
            N = int(STAT.split("ac")[1])
            sg["ret_next"] = sg.groupby(["instrument", "d"])["ret"].shift(-1)
            g = sg.assign(xy=sg["ret"] * sg["ret_next"], xx=sg["ret"] ** 2).groupby(["d", "instrument"])[["xy", "xx"]].sum()
            xy = g["xy"].unstack().reindex(days)
            xx = g["xx"].unstack().reindex(days)
            ac = (xy.rolling(N, min_periods=N // 2).sum() / xx.rolling(N, min_periods=N // 2).sum()).shift(1)
            s0 = ac.loc[out_days].stack().rename("signal").reset_index()
            for t in SLOTS:
                s = s0.copy()
                s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
                rows.append(s[["date", "instrument", "signal"]])
        else:                                                                      # T_acc
            prev_amt = sg.groupby("instrument")["amt"].shift(1)
            with np.errstate(all="ignore"):
                sg["signal"] = np.log(sg["amt"] / prev_amt.where(prev_amt > 0)) * np.sign(sg["ret"])
            s = sg[sg["seg"].isin(SLOTS) & sg["d"].isin(out_days)].copy()
            s["date"] = s["d"] + pd.to_timedelta(s["seg"].str[:2].astype(int), "h") + pd.to_timedelta(s["seg"].str[2:].astype(int), "m")
            rows.append(s[["date", "instrument", "signal"]])
        df = pd.concat(rows, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
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
    import pandas as pd
    from bigmodule import M
    import structlog

    logger = structlog.get_logger()
    datasources = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
    start_date = "2020-01-01 00:00:00"
    end_date = "2020-03-01 23:59:59"
    logger.info(f"计算因子，区间：{start_date} ~ {end_date}")
    factor_data = main(datasources, start_date, end_date)
    result = M.jyc_eval._latest(factor_data=factor_data, show=True)
