# -*- coding: utf-8 -*-
"""macstudio 上的 dai 替身：读本地 1 分钟 feather（价格 ÷100 换成元），按 filters 限定日期后执行因子 SQL。
同一 (SQL, 区间) 的结果在进程内缓存，同族变体只执行一次 SQL。只给 jobs/fac3 的批量研究用，不是平台环境。"""
import os
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather

ROOT = os.path.expanduser("~/Desktop/work/quant")
D1M = os.path.join(ROOT, "数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m")
COLS = ["date", "instrument_id", "open", "high", "low", "close", "deal_number", "volume", "amount",
        "ask_price1", "bid_price1", "ask_volume1", "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3",
        "ask_num_orders1", "ask_num_orders2", "ask_num_orders3", "bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]
PX = {"open", "high", "low", "close", "ask_price1", "bid_price1"}
_TBL = {"span": None, "tbl": None}
_CACHE = {}
_POOL = None
_CON = duckdb.connect()
_CON.execute("SET threads=8")


def load_span(sd, ed):
    """载入 [sd, ed] 覆盖的月份（只取 COLS），替换当前表。"""
    ms = pd.period_range(pd.Timestamp(sd).to_period("M"), pd.Timestamp(ed).to_period("M"), freq="M")
    parts = [feather.read_table(os.path.join(D1M, f"{m.strftime('%Y%m')}.0.feather"), columns=COLS)
             for m in ms if os.path.exists(os.path.join(D1M, f"{m.strftime('%Y%m')}.0.feather"))]
    _TBL["tbl"] = pa.concat_tables(parts)
    _TBL["span"] = (sd, ed)
    _CACHE.clear()
    _CON.register("_raw", _TBL["tbl"])
    sel = ", ".join((f"CAST({c} AS DOUBLE) / 100.0 AS {c}" if c in PX else "CAST(date AS TIMESTAMP) AS date" if c == "date" else c) for c in COLS)
    _CON.execute(f"CREATE OR REPLACE TEMP VIEW _bars AS SELECT {sel} FROM _raw")
    _CON.execute("CREATE OR REPLACE TEMP VIEW all_instruments AS SELECT DISTINCT instrument_id, "
                 "'ID' || lpad(CAST(instrument_id AS VARCHAR), 4, '0') AS instrument FROM _raw")


class _R(object):
    def __init__(self, df):
        self._df = df

    def df(self):
        return self._df.copy()


def query(sql, filters=None, **kw):
    global _POOL
    sd, ed = (filters or {}).get("date", [None, None])
    ed = str(ed) if len(str(ed)) > 10 else str(ed) + " 23:59:59"
    if "cpt_jyc_2026_instruments" in sql:
        if _POOL is None:
            _POOL = pd.read_parquet(os.path.join(ROOT, "e2e/cache/labels_y_off.parquet"), columns=["date", "instrument"])
        p = _POOL[(_POOL["date"] >= pd.Timestamp(sd)) & (_POOL["date"] <= pd.Timestamp(ed))]
        return _R(p)
    key = (sql, str(sd), ed)
    if key not in _CACHE:
        _CON.execute(f"CREATE OR REPLACE TEMP VIEW cpt_jyc_2026_stock_bar1m AS SELECT * FROM _bars "
                     f"WHERE date >= TIMESTAMP '{pd.Timestamp(sd)}' AND date <= TIMESTAMP '{pd.Timestamp(ed)}'")
        _CACHE[key] = _CON.execute(sql.replace("{bar1m}", "cpt_jyc_2026_stock_bar1m")).df()
    return _R(_CACHE[key])
