# -*- coding: utf-8 -*-
"""本地模拟 dai.query：只支持提交代码里用到的三类查询，底层读 Feather（伪代码 ID0001）。仅用于测试平台代码路径。"""
import re

import pandas as pd

from feather_data import FeatherBackend

_fb = FeatherBackend()
LOG = []


BOOK = ["bid_price1", "bid_price2", "bid_price3", "ask_price1", "ask_price2", "ask_price3",
        "bid_volume1", "bid_volume2", "bid_volume3", "ask_volume1", "ask_volume2", "ask_volume3",
        "bid_num_orders1", "bid_num_orders2", "bid_num_orders3", "ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]


def _run_win30_sql(sql, sd, ed):
    """平台 windows30 SQL 的本地执行：bar1m（含推导的昨收）与 all_instruments 经 parquet 交给 DuckDB（避开 numpy 2 接口）。"""
    import os
    import tempfile
    import duckdb
    import feather_data as F
    pool = _fb.pool(sd, ed)
    if not pool:                                # 本地数据之前的区间：平台上同样是空表
        return pd.DataFrame(columns=["date", "instrument"] + list(F.WIN30_COLS))
    old_cols = list(F.READ_COLS)                # 窗口 SQL 还要用到 open、deal_number（E07 S4 起）
    F.set_read_cols(list(set(old_cols) | {"open", "high", "low", "close", "volume", "amount", "deal_number"} | set(BOOK)))
    try:
        d = F.read_range(pd.Timestamp(sd) - pd.Timedelta(days=10), ed, pool)
    finally:
        F.set_read_cols(old_cols)
    d = d.sort_values(["instrument", "date"]).reset_index(drop=True)
    d["day"] = d["date"].dt.normalize()
    last = d.groupby(["instrument", "day"]).agg(dc=("close", "last"), da=("adjust_factor", "last"))
    g = last.groupby(level=0)
    last["pre_close"] = g["dc"].shift(1) * g["da"].shift(1) / last["da"]
    d = d.merge(last[["pre_close"]].reset_index(), on=["instrument", "day"], how="left")
    d = d[(d["date"] >= pd.Timestamp(sd)) & (d["date"] <= pd.Timestamp(ed))]
    d["instrument_id"] = d["instrument"].str[2:].astype(int)
    tmp = tempfile.mkdtemp(prefix="fake_dai_")
    d[["date", "instrument_id", "open", "high", "low", "close", "pre_close", "volume", "amount",
       "deal_number"] + BOOK].to_parquet(
        os.path.join(tmp, "bar1m.parquet"), index=False)
    d[["instrument_id", "instrument"]].drop_duplicates().to_parquet(os.path.join(tmp, "inst.parquet"), index=False)
    con = duckdb.connect()
    con.execute(f"CREATE VIEW bar1m AS SELECT * FROM read_parquet('{tmp}/bar1m.parquet')")
    con.execute(f"CREATE VIEW all_instruments AS SELECT * FROM read_parquet('{tmp}/inst.parquet')")
    table = re.search(r"FROM\s+(\w+)", sql)                         # 第一个 FROM 是 1 分钟表（窗口 SQL 与盘口 SQL 都是）
    q = sql.replace(f"FROM {table.group(1)}", "FROM bar1m", 1)
    cur = con.execute(q)
    return pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])


class _R(object):
    def __init__(self, df):
        self._df = df

    def df(self):
        return self._df.copy()


def query(sql, filters=None, **kwargs):  # 平台的 compression 等参数不影响结果
    filters = filters or {}
    sd, ed = filters["date"]
    if len(ed) == 10:
        ed = ed + " 23:59:59"
    inst = filters.get("instrument")
    if "cte_window" in sql:                     # 30 分钟窗口表：用 DuckDB 0.9 在本地 1 分钟数据上原样执行平台 SQL
        LOG.append(("book30" if "b_imb" in sql else "windows30", sd, ed, None))
        return _R(_run_win30_sql(sql, sd, ed))
    table = re.search(r"FROM\s+(\w+)", sql).group(1)
    LOG.append((table, sd, ed, None if inst is None else len(inst)))
    if table == "cpt_jyc_2026_instruments":
        pool = inst if inst is not None else _fb.pool(sd, ed)
        df = _fb.sections(sd, ed, pool)
        if "DISTINCT" in sql:
            df = df[["instrument"]].drop_duplicates()
        return _R(df)
    if table == "cpt_jyc_2026_exposure":        # 2026-10-12 起端到端线上环境不再提供该表：E07 起的包读它必须失败
        raise RuntimeError("端到端线上环境不提供 cpt_jyc_2026_exposure（2026-10-12 主办方通知）")
    if table == "cpt_jyc_2026_vwap":
        return _R(_fb.labels(sd, ed, inst).rename(columns={"y": "vwap_return"})[["date", "instrument", "vwap_return"]]
                  .rename(columns={"vwap_return": "y"} if "AS y" in sql else {}))
    cols = [c.strip() for c in re.search(r"SELECT\s+(.*?)\s+FROM", sql, re.S).group(1).split(",")]
    return _R(_fb.bars(sd, ed, inst)[cols])
