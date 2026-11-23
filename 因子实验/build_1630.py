# -*- coding: utf-8 -*-
"""由 1617 生成"1617 + 换手后处理（+ 时间锚点）"的交付代码（10-21 晚，用户："换手不太行"；今晚 10 个名额）。
1617 的计算原样不动，只在 main 返回之前加 _post_turnover：
  1）1617 输出逐截面 z 分数；
  2）可选加 w·z(上午相对 10:00 的回吐)：10:30／11:00／11:30 截面上 −ln(现价 / 10:00 价)，其余截面为 0
     （平台口径残差：剔除 1617 后 IC 0.0226、换手 0.34，是对 1617 增量最大的信号；口径同评分机 s10_am_drift：段末中间价，无则收盘价）；
  3）可选日内 EWMA（过去的权重 α，每个交易日重置；评分机的换手只算同一交易日相邻截面）；
  4）可选缓冲带：x_t = z_t + β·(1[上一截面同日在多头腿] − 1[在空头腿])，腿 = 排名 > 0.9／≤ 0.1，按处理后的值判定（迟滞）；
  5）09:30 置 0。
用法：python3 因子实验/build_1630.py <编号> <名字> <ew> <buf> <anchor_w> [<q_w> [<q_mode both|s1> [<q_sm>]]]
例：python3 因子实验/build_1630.py 1630 1617日内平滑0.25 0.25 0 0"""
import sys

import os
BASES = {"1617": ("因子代码/1617_非线性堆叠十六模型平均深度四五多期目标平台口径.py", "【1617：非线性堆叠十六模型平均深度四五多期目标平台口径】", "1617（公榜 0.90451）"),
         "1626": ("因子代码/1626_非线性堆叠十六模型平均全量拟合至2024.py", "【1626：非线性堆叠十六模型平均全量拟合至2024】", "1626（公榜 0.90287；1617 配置全量拟合至 2024）")}
BASE = os.environ.get("BASE", "1617")                                         # 10-23：底座可换成 1626（同配置的树拟合到 2024）
SRC, ANCHOR, BLABEL = BASES[BASE]
num, title, ew, buf, aw = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
qw = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0                      # 最后 10 分钟委托结构的权重（10-21 晚加）
qmode = sys.argv[7] if len(sys.argv) > 7 else "both"                          # both = 深档笔数 + 一档规模比；s1 = 只用一档规模比
qsm = float(sys.argv[8]) if len(sys.argv) > 8 else 0.3                        # 委托结构的跨截面平滑（过去的权重）
DST = "因子代码/%s_%s.py" % (num, title)
s = open(SRC, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:80])
    s = s.replace(a, b)


parts = []
if aw:
    parts.append("加 %.2f·上午相对 10:00 的回吐（10:30–11:30，−ln(现价 / 10:00 价)；平台口径剔除 1617 后 IC 0.0226、换手 0.34）" % aw)
if qw:
    parts.append("加 %.2f·最后 10 分钟委托结构（%s，截面前 10 分钟均值，平滑 %.1f）" % (qw, "深档笔数失衡 + 一档委托规模比取负" if qmode == "both" else "一档委托规模比取负", qsm))
if ew:
    parts.append("日内 EWMA α=%.2f（过去的权重，每日重置）" % ew)
if buf:
    parts.append("腿边界缓冲带 β=%.2f（上一截面同日在多头／空头腿的股票加／减 β 个标准差，迟滞）" % buf)
desc = "；".join(parts)
rep('''    """''' + ANCHOR,
    '''    """【%s：%s】= %s原样计算，返回前加换手后处理：逐截面 z 分数；%s；09:30 置 0。
      动机：评分机的换手 = 多头腿（排名 > 0.9）／空头腿（≤ 0.1）在同一交易日相邻截面间的替换比例；1617 换手 0.52（1610 是 0.38），因子榜对换手罚得很重。
    以下为 %s 的原说明。
    ''' % (num, title, BLABEL, desc, BASE) + ANCHOR)
rep('''    out = pd.DataFrame({"date": key["date"].to_numpy(), "instrument": key["instrument"].to_numpy(), "factor": np.nan_to_num(v, nan=0.0)})
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)''',
    '''    out = pd.DataFrame({"date": key["date"].to_numpy(), "instrument": key["instrument"].to_numpy(), "factor": np.nan_to_num(v, nan=0.0)})
    out = _post_turnover(out, datasources, start_date, end_date)
    return out.sort_values(["date", "instrument"]).reset_index(drop=True)''')
# 新函数插在 __main__ 自测块之前（10-19 教训：追加在自测块之后的函数会被 make_ipynb 之外的流程漏掉，平台报 NameError）
_cut = s.index('\n\nif __name__ == "__main__":')
_tail = s[_cut:]
s = s[:_cut].rstrip("\n") + '''


_POST_EW, _POST_BUF, _POST_ANCHOR_W, _POST_Q_W, _POST_Q_MODE, _POST_Q_SM = %r, %r, %r, %r, %r, %r


def _post_turnover(out, datasources, start_date, end_date):
    """换手后处理：逐截面 z 分数 →（加时间锚点）→ 日内 EWMA → 腿边界缓冲带 → 09:30 置 0。只用截面时刻及之前的数据。"""
    import numpy as np
    import pandas as pd

    d = out[["date", "instrument", "factor"]].copy()
    d["date"] = pd.to_datetime(d["date"])
    d["instrument"] = d["instrument"].astype(str)
    hm = d["date"].dt.strftime("%%H%%M")
    z930 = (hm == "0930").to_numpy()

    def _z(x):
        g = x.groupby(d["date"])
        return ((x - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)

    d["z"] = _z(pd.to_numeric(d["factor"], errors="coerce").fillna(0.0))
    d.loc[z930, "z"] = 0.0
    if _POST_ANCHOR_W > 0:
        a = _am_drift(datasources, start_date, end_date)
        d = d.merge(a, on=["date", "instrument"], how="left")
        am = d["am"].astype(float)
        za = _z((am.groupby(d["date"]).rank(pct=True) - 0.5).fillna(0.0))        # 与评分机任务 s10_am_drift 同口径：先逐截面排名再标准化
        za[~hm.isin(["1030", "1100", "1130"]).to_numpy()] = 0.0
        d["z"] = d["z"] + _POST_ANCHOR_W * za
        d = d.drop(columns=["am"])
        d["z"] = _z(d["z"])
    if _POST_Q_W > 0:
        q = _q_late(datasources, start_date, end_date)
        d = d.merge(q, on=["date", "instrument"], how="left")
        d["z"] = _z(d["z"] + _POST_Q_W * _z(d["q"].astype(float).fillna(0.0)))
        d = d.drop(columns=["q"])
    d["day"] = d["date"].dt.normalize()
    if _POST_EW > 0:
        d = d.sort_values(["instrument", "date"]).reset_index(drop=True)
        v, ins, day = d["z"].to_numpy(float), d["instrument"].to_numpy(), d["day"].to_numpy()
        is930 = (d["date"].dt.strftime("%%H%%M") == "0930").to_numpy()
        res = np.empty_like(v)
        prev, li, ld = 0.0, None, None
        for i in range(len(v)):
            if is930[i]:                                                # 09:30 不参与（全为 0），当天从 10:00 开始
                res[i], li = 0.0, None
                continue
            if ins[i] != li or day[i] != ld:
                prev = v[i]
            else:
                prev = _POST_EW * prev + (1 - _POST_EW) * v[i]
            res[i], li, ld = prev, ins[i], day[i]
        d["z"] = res
        d["z"] = _z(d["z"])
    if _POST_BUF > 0:
        d = d.sort_values(["date", "instrument"]).reset_index(drop=True)
        res = d["z"].to_numpy(float).copy()
        prev_leg, prev_day = {}, None
        for _, idx in d.groupby("date", sort=True).indices.items():
            sub = d.iloc[idx]
            dd = sub["day"].iloc[0]
            if sub["date"].iloc[0].strftime("%%H%%M") == "0930":
                prev_leg, prev_day = {}, dd
                continue
            vv = res[idx]
            if prev_day is not None and dd == prev_day and prev_leg:
                vv = vv + _POST_BUF * sub["instrument"].map(prev_leg).fillna(0.0).to_numpy(float)
            rr = pd.Series(vv).rank(pct=True).to_numpy()
            res[idx] = vv
            prev_leg = dict(zip(sub["instrument"].to_numpy(), np.where(rr > 0.9, 1.0, np.where(rr <= 0.1, -1.0, 0.0))))
            prev_day = dd
        d["z"] = res
    d.loc[d["date"].dt.strftime("%%H%%M") == "0930", "z"] = 0.0
    return pd.DataFrame({"date": d["date"].to_numpy(), "instrument": d["instrument"].to_numpy(), "factor": d["z"].astype("float64").fillna(0.0).to_numpy()})


def _q_late(datasources, start_date, end_date):
    """最后 10 分钟委托结构（与评分机任务 s3_q_mix_late 同一口径）：每个半小时段最后 10 分钟内各分钟的
    深档（2、3 档）委托笔数失衡 (买 − 卖)/(买 + 卖) 的均值、一档平均委托规模比 log[(买一量/买一笔数)/(卖一量/卖一笔数)] 的均值；
    逐截面排名（缺失记 0.5）后相加（_POST_Q_MODE = "s1" 时只用一档规模比取负），再在股票池内逐截面排名 − 0.5（缺失记 0），
    按股票跨截面 EWMA（过去的权重 _POST_Q_SM），09:30 置 0。
    向前多取 10 个自然日做平滑的预热。只用截面时刻及之前的数据。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    q0 = (pd.to_datetime(start_date) - pd.Timedelta(days=10)).strftime("%%Y-%%m-%%d %%H:%%M:%%S")
    sql = f"""
    WITH b AS (
        SELECT b.instrument_id, ai.instrument, CAST(b.date AS DATE) AS d, strftime(b.date, '%%H%%M%%S') AS hms,
            CASE
                WHEN strftime(b.date, '%%H%%M%%S') > '093000' AND strftime(b.date, '%%H%%M%%S') <= '100000' THEN '1000'
                WHEN strftime(b.date, '%%H%%M%%S') > '100000' AND strftime(b.date, '%%H%%M%%S') <= '103000' THEN '1030'
                WHEN strftime(b.date, '%%H%%M%%S') > '103000' AND strftime(b.date, '%%H%%M%%S') <= '110000' THEN '1100'
                WHEN strftime(b.date, '%%H%%M%%S') > '110000' AND strftime(b.date, '%%H%%M%%S') <= '113000' THEN '1130'
                WHEN strftime(b.date, '%%H%%M%%S') > '130000' AND strftime(b.date, '%%H%%M%%S') <= '133000' THEN '1330'
                WHEN strftime(b.date, '%%H%%M%%S') > '133000' AND strftime(b.date, '%%H%%M%%S') <= '140000' THEN '1400'
                WHEN strftime(b.date, '%%H%%M%%S') > '140000' AND strftime(b.date, '%%H%%M%%S') <= '143000' THEN '1430'
            END AS seg,
            CAST(COALESCE(b.bid_num_orders2, 0) + COALESCE(b.bid_num_orders3, 0) AS DOUBLE) AS bn23,
            CAST(COALESCE(b.ask_num_orders2, 0) + COALESCE(b.ask_num_orders3, 0) AS DOUBLE) AS an23,
            CAST(COALESCE(b.bid_num_orders1, 0) AS DOUBLE) AS bn1, CAST(COALESCE(b.ask_num_orders1, 0) AS DOUBLE) AS an1,
            CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1, CAST(COALESCE(b.ask_volume1, 0) AS DOUBLE) AS av1
        FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE strftime(b.date, '%%H%%M%%S') > '093000' AND strftime(b.date, '%%H%%M%%S') <= '143000'
    ),
    l AS (
        SELECT *, hms > strftime(strptime(seg || '00', '%%H%%M%%S') - INTERVAL 10 MINUTE, '%%H%%M%%S') AS late FROM b WHERE seg IS NOT NULL
    )
    SELECT MAX(instrument) AS instrument, d, seg,
        AVG(CASE WHEN late AND bn23 + an23 > 0 THEN (bn23 - an23) / (bn23 + an23) END) AS qn23l,
        AVG(CASE WHEN late AND bn1 > 0 AND an1 > 0 AND bv1 > 0 AND av1 > 0 THEN ln((bv1 / bn1) / (av1 / an1)) END) AS qs1l
    FROM l GROUP BY instrument_id, d, seg
    """
    sg = dai.query(sql, filters={"date": [q0, end_date]}, compression=True).df()
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}", filters={"date": [q0, end_date]}, compression=True).df()
    pool["date"] = pd.to_datetime(pool["date"])
    pool["instrument"] = pool["instrument"].astype(str)
    pool = pool.drop_duplicates(["date", "instrument"])
    if not len(sg):
        return pd.DataFrame(columns=["date", "instrument", "q"])
    sg = sg.dropna(subset=["instrument"])
    sg["instrument"] = sg["instrument"].astype(str)
    sg["date"] = pd.to_datetime(sg["d"]) + pd.to_timedelta(sg["seg"].str[:2].astype(int), unit="h") + pd.to_timedelta(sg["seg"].str[2:].astype(int), unit="m")
    sg = sg.drop_duplicates(["date", "instrument"])
    gq = sg.groupby("date")
    if _POST_Q_MODE == "s1":                                                    # 只用一档委托规模比（取负）
        sg["x"] = -sg["qs1l"]
    else:
        sg["x"] = gq["qn23l"].rank(pct=True).fillna(0.5) + (-sg["qs1l"]).groupby(sg["date"]).rank(pct=True).fillna(0.5)
    res = pool.merge(sg[["date", "instrument", "x"]], on=["date", "instrument"], how="left")
    res["q"] = (res.groupby("date")["x"].rank(pct=True) - 0.5).fillna(0.0)
    hm = res["date"].dt.strftime("%%H%%M")
    res.loc[hm == "0930", "q"] = 0.0
    res = res.sort_values(["instrument", "date"])
    res["q"] = res.groupby("instrument", sort=False)["q"].transform(lambda s_: s_.ewm(alpha=1.0 - _POST_Q_SM, adjust=False).mean())
    res.loc[res["date"].dt.strftime("%%H%%M") == "0930", "q"] = 0.0
    res = res[(res["date"] >= pd.to_datetime(start_date)) & (res["date"] <= pd.to_datetime(end_date))]
    return res[["date", "instrument", "q"]]


def _am_drift(datasources, start_date, end_date):
    """上午相对 10:00 的回吐：10:30／11:00／11:30 截面上 −ln(段末价 / 10:00 段末价)；段末价 = 该段最后一分钟的中间价（买一、卖一有效时），否则收盘价。"""
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    sql = f"""
    WITH b AS (
        SELECT b.instrument_id, ai.instrument, CAST(b.date AS DATE) AS d, strftime(b.date, '%%H%%M%%S') AS hms,
            CASE
                WHEN strftime(b.date, '%%H%%M%%S') > '093000' AND strftime(b.date, '%%H%%M%%S') <= '100000' THEN '1000'
                WHEN strftime(b.date, '%%H%%M%%S') > '100000' AND strftime(b.date, '%%H%%M%%S') <= '103000' THEN '1030'
                WHEN strftime(b.date, '%%H%%M%%S') > '103000' AND strftime(b.date, '%%H%%M%%S') <= '110000' THEN '1100'
                WHEN strftime(b.date, '%%H%%M%%S') > '110000' AND strftime(b.date, '%%H%%M%%S') <= '113000' THEN '1130'
            END AS seg,
            CASE WHEN CAST(b.bid_price1 AS DOUBLE) > 0 AND CAST(b.ask_price1 AS DOUBLE) > CAST(b.bid_price1 AS DOUBLE)
                 THEN (CAST(b.bid_price1 AS DOUBLE) + CAST(b.ask_price1 AS DOUBLE)) / 2.0
                 WHEN CAST(b.close AS DOUBLE) > 0 THEN CAST(b.close AS DOUBLE) END AS px
        FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE strftime(b.date, '%%H%%M%%S') > '093000' AND strftime(b.date, '%%H%%M%%S') <= '113000'
    )
    SELECT MAX(instrument) AS instrument, d, seg, arg_max(px, hms) AS px
    FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
    """
    sg = dai.query(sql, filters={"date": [start_date, end_date]}, compression=True).df()
    if not len(sg):
        return pd.DataFrame(columns=["date", "instrument", "am"])
    sg = sg.dropna(subset=["instrument"])
    sg["instrument"] = sg["instrument"].astype(str)
    sg["d"] = pd.to_datetime(sg["d"])
    p10 = sg[sg["seg"] == "1000"][["instrument", "d", "px"]].rename(columns={"px": "p10"})
    sg = sg[sg["seg"] != "1000"].merge(p10, on=["instrument", "d"], how="left")
    with np.errstate(all="ignore"):
        sg["am"] = -np.log(sg["px"].where(sg["px"] > 0) / sg["p10"].where(sg["p10"] > 0))
    sg["date"] = sg["d"] + pd.to_timedelta(sg["seg"].str[:2].astype(int), unit="h") + pd.to_timedelta(sg["seg"].str[2:].astype(int), unit="m")
    return sg[["date", "instrument", "am"]].replace([np.inf, -np.inf], np.nan).drop_duplicates(["date", "instrument"])
''' % (ew, buf, aw, qw, qmode, qsm) + _tail
assert s.count('if __name__ == "__main__":') == 1 and s.index("def _post_turnover") < s.index('if __name__ == "__main__":')
open(DST, "w", encoding="utf-8").write(s)
print("写出", DST)
