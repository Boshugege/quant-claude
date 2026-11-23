# -*- coding: utf-8 -*-
"""由 1610 生成 1628（1610 加委托结构）：在 1610 的"A + Σ w × (B 的截面排名 − 0.5)"框架里加第 13 根 B13 = 委托结构。
  B13 = 平均(深档笔数失衡的截面排名, −一档平均委托规模比的截面排名)，先做 EWMA α=0.3（过去的权重），权重 0.2234。
  权重 = 0.2 × σ(1610) ÷ σ(均匀排名)：c98／b5 里"1610/σ + 0.2·委托结构/σ"的等价写法（σ(1610) = 0.3225，2024 非 09:30 行）。
  口径同 因子实验/m15_queue.py（OQ_SIZE）与 m17_queue2.py（QL_NIMB23）：分钟级指标在 (段起点, 段终点] 内取均值。
用法：python3 因子实验/build_1628.py"""
SRC = "因子代码/1610_队友161头部加七根三小根收盘段季节性与收盘位置锚.py"
DST = "因子代码/1628_1610加委托结构.py"
s = open(SRC, encoding="utf-8").read()


def rep(a, b, n=1):
    global s
    assert s.count(a) == n, (s.count(a), a[:80])
    s = s.replace(a, b)


rep('''    """【1610：队友161头部加七根三小根收盘段季节性与收盘位置锚】= 1602（公榜 0.86667）的升级版：''',
    '''    """【1628：1610 加委托结构】= 1610（公榜 0.891）原样，加第 13 根 B13 委托结构（盘口排队的人数与挂单规模）：
      B13 × 0.2234（先做 EWMA α=0.3）：委托结构 = 平均(深档（2、3 档）委托笔数失衡 (买 − 卖)/(买 + 卖) 的截面排名,
        −一档平均单笔挂单规模之比 log[(买一量/买一笔数)/(卖一量/卖一笔数)] 的截面排名)，两项都是截面所在半小时段内各分钟的均值。
        含义：排在后面（2、3 档）的买方队伍人数多于卖方 → 随后上涨；一档买方是少数大单托底、卖方是很多小单 → 随后下跌（大单挂在哪边，价格就往反方向走）。
      本地相对 1610：样本外 2023／2024 两窗五项全好（IC +0.0012～+0.0013、IC_IR +0.018、压力 +0.020～+0.025、多空 +0.15～+0.48、换手 −0.0005～−0.0034）；
      评分机 2024：IC 0.0720 / IC_IR 1.279 / 多空 38.40 / 压力 1.219 / 换手 0.374（1610：0.0708 / 1.261 / 38.31 / 1.199 / 0.374）。
    以下为 1610 的原说明。
    【1610：队友161头部加七根三小根收盘段季节性与收盘位置锚】= 1602（公榜 0.86667）的升级版：''')
rep('''      B12 × 0.072：收盘位置锚（收盘÷当日半小时收盘最高价−1 的 20 日均值，全天同值）
''', '''      B12 × 0.072：收盘位置锚（收盘÷当日半小时收盘最高价−1 的 20 日均值，全天同值）
      B13 × 0.2234（先做 EWMA α=0.3）：委托结构（1628 新增，见上）
''')
rep('''        (main_b12, 0.072, 0.0, {}),
    ]:''', '''        (main_b12, 0.072, 0.0, {}),
        (main_b13, 0.2234, 0.3, {}),
    ]:''')
assert '__main__' not in s
s = s.rstrip("\n") + '''


def main_b13(datasources, start_date, end_date):
    """【委托结构 B13】平均(深档（2、3 档）委托笔数失衡的截面排名, −一档平均单笔挂单规模之比的截面排名)。
    分钟级：深档笔数失衡 = (买二 + 买三笔数 − 卖二 − 卖三笔数) ÷ 四者之和；一档规模比 = log[(买一量 ÷ 买一笔数) ÷ (卖一量 ÷ 卖一笔数)]（四个值都为正时才算）。
    截面 t 的取值 = (t − 30 分钟, t] 内各分钟的均值（10:00 截面用 09:31～10:00），只用截面时刻及之前的数据。
    各自在股票池内逐截面百分位排名 − 0.5（取值为 0 或缺失记 0），两项平均；09:30 置 0。按自然月分批调用，控制内存。
    """
    import pandas as pd
    parts = []
    for _ms in pd.period_range(pd.Timestamp(start_date).to_period("M"), pd.Timestamp(end_date).to_period("M"), freq="M"):
        _s = max(pd.Timestamp(start_date), _ms.start_time)
        _e = min(pd.Timestamp(end_date), _ms.end_time)
        if _s > _e:
            continue
        parts.append(_q_impl_b13(datasources, _s.strftime("%Y-%m-%d %H:%M:%S"), _e.strftime("%Y-%m-%d %H:%M:%S")))
    out = pd.concat(parts, ignore_index=True)
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def _q_impl_b13(datasources, start_date, end_date):
    import numpy as np
    import pandas as pd
    import dai

    bar1m = datasources["bar1m"]
    sql_q = f"""
    WITH b AS (
        SELECT b.instrument_id, ai.instrument, CAST(b.date AS DATE) AS d,
            CASE
                WHEN strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '100000' THEN '1000'
                WHEN strftime(b.date, '%H%M%S') > '100000' AND strftime(b.date, '%H%M%S') <= '103000' THEN '1030'
                WHEN strftime(b.date, '%H%M%S') > '103000' AND strftime(b.date, '%H%M%S') <= '110000' THEN '1100'
                WHEN strftime(b.date, '%H%M%S') > '110000' AND strftime(b.date, '%H%M%S') <= '113000' THEN '1130'
                WHEN strftime(b.date, '%H%M%S') > '130000' AND strftime(b.date, '%H%M%S') <= '133000' THEN '1330'
                WHEN strftime(b.date, '%H%M%S') > '133000' AND strftime(b.date, '%H%M%S') <= '140000' THEN '1400'
                WHEN strftime(b.date, '%H%M%S') > '140000' AND strftime(b.date, '%H%M%S') <= '143000' THEN '1430'
            END AS seg,
            CAST(COALESCE(b.bid_num_orders2, 0) + COALESCE(b.bid_num_orders3, 0) AS DOUBLE) AS bn23,
            CAST(COALESCE(b.ask_num_orders2, 0) + COALESCE(b.ask_num_orders3, 0) AS DOUBLE) AS an23,
            CAST(COALESCE(b.bid_num_orders1, 0) AS DOUBLE) AS bn1, CAST(COALESCE(b.ask_num_orders1, 0) AS DOUBLE) AS an1,
            CAST(COALESCE(b.bid_volume1, 0) AS DOUBLE) AS bv1, CAST(COALESCE(b.ask_volume1, 0) AS DOUBLE) AS av1
        FROM {bar1m} b LEFT JOIN all_instruments ai USING (instrument_id)
        WHERE strftime(b.date, '%H%M%S') > '093000' AND strftime(b.date, '%H%M%S') <= '143000'
    )
    SELECT instrument_id, MAX(instrument) AS instrument, d, seg,
        AVG(CASE WHEN bn23 + an23 > 0 THEN (bn23 - an23) / (bn23 + an23) END) AS n23,
        AVG(CASE WHEN bn1 > 0 AND an1 > 0 AND bv1 > 0 AND av1 > 0 THEN ln((bv1 / bn1) / (av1 / an1)) END) AS s1
    FROM b WHERE seg IS NOT NULL GROUP BY instrument_id, d, seg
    """
    sg = dai.query(sql_q, filters={"date": [start_date, end_date]}, compression=True).df()
    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%H%M").isin(_slots)].drop_duplicates(["date", "instrument"])
    if not len(sg):
        _pool["factor"] = 0.0
        return _pool[["date", "instrument", "factor"]]
    sg = sg.dropna(subset=["instrument"])
    sg["instrument"] = sg["instrument"].astype(str)
    sg["date"] = pd.to_datetime(sg["d"]) + pd.to_timedelta(sg["seg"].str[:2].astype(int), unit="h") + pd.to_timedelta(sg["seg"].str[2:].astype(int), unit="m")
    sg = sg.drop_duplicates(["date", "instrument"])
    out = _pool.merge(sg[["date", "instrument", "n23", "s1"]], on=["date", "instrument"], how="left")

    def _rk(x):
        x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)
        x = x.where(x != 0)
        return (x.groupby(out["date"]).rank(pct=True) - 0.5).fillna(0.0)

    out["factor"] = 0.5 * (_rk(out["n23"]) + _rk(-out["s1"]))
    out["factor"] = out["factor"].astype("float64").fillna(0.0)
    out.loc[out["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)
'''
open(DST, "w", encoding="utf-8").write(s)
print("写出", DST)
