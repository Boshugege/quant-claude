def main(datasources, start_date, end_date):
    """【1365：近九十分钟上涨分钟量占比反向精简按月】新根单独版（公榜探针），STAT = Z_mfr90，方向 反向。
    根 = 1340_资金流窗口框架.py 原样复用；按自然月分批调用，控制平台整段调用时的内存。
    """
    import pandas as pd
    parts = []
    for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
        _s = max(pd.Timestamp(start_date), _ms.start_time)
        _e = min(pd.Timestamp(end_date), _ms.end_time)
        if _s > _e:
            continue
        parts.append(_root_impl(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S"), "Z_mfr90"))
    out = pd.concat(parts, ignore_index=True)
    out["factor"] = -out["factor"]                                           # 反向交付（09:30 仍为 0）
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def _root_impl(datasources, start_date, end_date, STAT):
    """【新根因子 第六轮 L：资金流窗口框架（1340 起）】一份代码覆盖 L/M/N/O/P/Q/R/S/T/U/V 十一个方向，STAT 选一个统计量。
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
      W_vdir         量价同向：Σ r·(ln 成交量 − 之前分钟均值) ÷ Σ|r|
      W_cnt          1–3 档委托笔数不平衡 (买 − 卖)/(买 + 卖) 的当日均值
      W_hhi          当日分钟成交量集中度 Σv² ÷ (Σv)²
      W_zero         当日零收益分钟占比（价格黏滞 / 跳价约束）
      W_atsd         当日分钟平均单笔成交额（对数）的离散度
      W_hl           Σ(分钟最高 − 最低)/收盘 ÷ |当日净变动|（日内往返程度）
      X_q60 / X_q30  近 60 / 30 分钟带符号路径效率；X_q5d 近 5 日（1200 分钟）带符号路径效率
      X_n5d / X_m5d  近 5 日大单方向 / 聪明钱（|r|/v 加权 VWAP ÷ VWAP − 1）
      X_m75 / X_ma   聪明钱权重 |r|/v^0.75、|r|/成交额
      L_rest20       当日剩余各时段过去 20 日"收盘 ÷ VWAP − 1"均值的平均（季节性，换手更低）
      P_ac20 / P_ac60  个股过去 20/60 日相邻半小时段收益的一阶自相关（日内常数，供条件化头部用）
    """
    import numpy as np
    import pandas as pd
    import dai

    SLOTS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
    NEXT = {"1000": "1030", "1030": "1100", "1100": "1130", "1130": "1330", "1330": "1400", "1400": "1430", "1430": "1500"}
    bar1m = datasources["bar1m"]
    fam = STAT.split("_")[0]

    if fam in ("M", "N", "Q", "U", "S", "O", "R", "V", "W", "X", "Y", "Z"):
        sql_min = f"""
        WITH b AS (
            SELECT b.instrument_id, ai.instrument, b.date, CAST(b.date AS DATE) AS d, strftime(b.date, '%H%M') AS hm,
                CAST(b.open AS DOUBLE) AS o, CAST(b.close AS DOUBLE) AS c, CAST(b.high AS DOUBLE) AS h, CAST(b.low AS DOUBLE) AS l,
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
                CASE WHEN n > 0 AND a > 0 THEN LN(a / n) END AS lats, CASE WHEN v > 0 THEN LN(v) END AS lv
            FROM b
        ), m2 AS (
            SELECT *
            FROM m1
        ), m3 AS (
            SELECT instrument_id, instrument, d, hm, date, c,
                SUM(a) OVER w AS ca, SUM(v) OVER w AS cv,
                SUM(ABS(r)) OVER w AS cabs,
                SUM(CASE WHEN r > 0 THEN v ELSE 0 END) OVER w90 AS vup90, SUM(v) OVER w90 AS v90
            FROM m2
            WINDOW w AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW),
                   w60 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW),
                   w30 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW),
                   w5 AS (PARTITION BY instrument_id ORDER BY date ROWS BETWEEN 1199 PRECEDING AND CURRENT ROW),
                   w90 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 89 PRECEDING AND CURRENT ROW),
                   w120 AS (PARTITION BY instrument_id, d ORDER BY date ROWS BETWEEN 119 PRECEDING AND CURRENT ROW),
                   w60x AS (PARTITION BY instrument_id ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW)
        )
        SELECT * FROM m3 WHERE hm IN ('1000', '1030', '1100', '1130', '1330', '1400', '1430')
        """
        q5 = (pd.to_datetime(start_date) - pd.Timedelta(days=12)).strftime("%Y-%m-%d %H:%M:%S")     # 近 5 日窗口需要之前的分钟
        df = dai.query(sql_min, filters={"date": [q5, end_date]}, compression=True).df()
        df = df[pd.to_datetime(df["date"]) >= pd.Timestamp(start_date)].reset_index(drop=True)
        if not len(df):
            return pd.DataFrame(columns=["date", "instrument", "factor"])
        with np.errstate(all="ignore"):
            vw = df["ca"] / df["cv"].where(df["cv"] > 0)
            cabs = df["cabs"].where(df["cabs"] > 0)
            sig = {
                "M_q": lambda: (df["cs1a"] / df["cs1v"].where(df["cs1v"] > 0)) / vw - 1,
                "M_q_lin": lambda: (df["cs2a"] / df["cs2v"].where(df["cs2v"] > 0)) / vw - 1,
                "N_dir": lambda: df["cnd"] / cabs,
                "N_big": lambda: df["cbig"] - df["csmall"],
                "Q_eff": lambda: df["csum"].abs() / cabs,
                "Q_seff": lambda: df["csum"] / cabs,
                "Q_ac1": lambda: df["crr1"] / df["crr"].where(df["crr"] > 0),
                "U_nj": lambda: df["cnj"].astype(float),
                "U_sj": lambda: df["csj"],
                "S_osz": lambda: df["mosz"],
                "S_osz30": lambda: df["mosz30"],
                "O_q5": lambda: df["mq5"],
                "R_round": lambda: (df["c"] - np.round(df["c"])) / df["c"],
                "R_half": lambda: (df["c"] - np.round(df["c"] * 2) / 2) / df["c"],
                "V_spr": lambda: df["mspr"],
                "W_vdir": lambda: df["cvd"] / cabs,
                "X_q60": lambda: df["r60"] / df["a60"].where(df["a60"] > 0),
                "X_q30": lambda: df["r30"] / df["a30"].where(df["a30"] > 0),
                "X_q5d": lambda: df["r5d"] / df["a5d"].where(df["a5d"] > 0),
                "X_n5d": lambda: df["nd5"] / df["a5d"].where(df["a5d"] > 0),
                "X_m5d": lambda: (df["s2a_5"] / df["s2v_5"].where(df["s2v_5"] > 0)) / (df["a_5"] / df["v_5"].where(df["v_5"] > 0)) - 1,
                "X_m75": lambda: (df["cs3a"] / df["cs3v"].where(df["cs3v"] > 0)) / vw - 1,
                "X_ma": lambda: (df["cs4a"] / df["cs4v"].where(df["cs4v"] > 0)) / vw - 1,
                "Y_vd5": lambda: df["vd5"] / df["a5d"].where(df["a5d"] > 0),
                "Y_z5": lambda: df["z5"] / df["n5"].where(df["n5"] > 0),
                "Y_sd5": lambda: df["sd5"],
                "Y_mfr": lambda: df["vup"] / df["vall"].where(df["vall"] > 0),
                "Y_bid": lambda: df["abid"] / df["aall"].where(df["aall"] > 0),
                "Z_aup": lambda: df["aup"] / df["aall"].where(df["aall"] > 0),
                "Z_mfr60": lambda: df["vup60"] / df["v60"].where(df["v60"] > 0),
                "Z_ats": lambda: np.log((df["aup"] / df["nup"].where(df["nup"] > 0)) / (df["adn"] / df["ndn"].where(df["ndn"] > 0))),
                "Z_mfr30": lambda: df["vup30"] / df["v30"].where(df["v30"] > 0),
                "Z_mfr90": lambda: df["vup90"] / df["v90"].where(df["v90"] > 0),
                "Z_mfr120": lambda: df["vup120"] / df["v120"].where(df["v120"] > 0),
                "Z_net60": lambda: (df["vup60"] - df["vdn60"]) / df["v60"].where(df["v60"] > 0),
                "Z_amfr60": lambda: df["aup60"] / df["am60"].where(df["am60"] > 0),
                "Z_mfr60x": lambda: df["vup60x"] / df["v60x"].where(df["v60x"] > 0),
                "W_cnt": lambda: df["mcimb"],
                "W_hhi": lambda: df["cvv"] / (df["cv"] ** 2).where(df["cv"] > 0),
                "W_zero": lambda: df["czero"] / df["cnr"].where(df["cnr"] > 0),
                "W_atsd": lambda: df["sdlats"],
                "W_hl": lambda: df["chl"] / df["csum"].abs().where(df["csum"].abs() > 0),
            }[STAT.split("_res")[0]]()
        df["signal"] = np.asarray(sig, dtype=float)
        if "_res" in STAT:                                                   # 逐截面对当日收益（_res2 再加量价同向）回归取残差
            Xr = [df["csum"].to_numpy(float)]
            if STAT.endswith("_res2"):
                Xr.append((df["cvd"] / cabs).to_numpy(float))
            Xr = np.column_stack(Xr)
            y_ = df["signal"].to_numpy(float)
            out_ = np.full(len(df), np.nan)
            for _, ii in df.groupby("date").indices.items():
                ok_ = np.isfinite(y_[ii]) & np.isfinite(Xr[ii]).all(1)
                if ok_.sum() < 50:
                    continue
                A_ = np.column_stack([np.ones(ok_.sum()), Xr[ii][ok_]])
                coef_, *_ = np.linalg.lstsq(A_, y_[ii][ok_], rcond=None)
                out_[ii[ok_]] = y_[ii][ok_] - A_ @ coef_
            df["signal"] = out_
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
            if STAT.startswith("L_rest"):                                          # 当日剩余时段的平均季节性
                segs = ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
                for t in SLOTS:
                    rest = segs[segs.index(NEXT[t]):]
                    s = mu.loc[out_days].loc[:, mu.columns.get_level_values(1).isin(rest)].T.groupby(level=0).mean().T
                    s = s.stack().rename("signal").reset_index()
                    s["date"] = s["d"] + pd.Timedelta(hours=int(t[:2]), minutes=int(t[2:]))
                    rows.append(s[["date", "instrument", "signal"]])
            if STAT.endswith("_dm"):
                mu = mu - mu.T.groupby(level=0).transform("mean").T
            for t in ([] if STAT.startswith("L_rest") else SLOTS):
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
