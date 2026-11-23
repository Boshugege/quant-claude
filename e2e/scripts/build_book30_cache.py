# -*- coding: utf-8 -*-
"""本地盘口窗口表缓存：在本地 1 分钟数据上用 DuckDB（与平台同代）原样执行提交脚本的 _BOOK30_SQL，逐月写
e2e/cache/book30/YYYYMM.parquet（date, instrument, b_imb, b_spr, b_conc, b_oimb），供 FeatherBackend.book30 读取。
平台上同一段 SQL 直接查 bar1m 表，所以两边口径一致；比值类特征与价格 / 量的计量单位无关。
用法：python e2e/scripts/build_book30_cache.py [起始月 YYYYMM] [结束月 YYYYMM]
"""
import os
import sys

import duckdb
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import feather_data as F  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402

OUT = os.path.join(HERE, "..", "cache", "book30v3")      # 第三版：多 b_mp、b_dep（reports/15 T4）；旧列与 book30/ 逐位相同
COLS = ["bid_price1", "ask_price1", "bid_volume1", "bid_volume2", "bid_volume3", "ask_volume1", "ask_volume2", "ask_volume3",
        "bid_num_orders1", "bid_num_orders2", "bid_num_orders3", "ask_num_orders1", "ask_num_orders2", "ask_num_orders3", "volume"]
months = sorted(f.split(".")[0] for f in os.listdir(F.DATA_1M) if f.endswith(".feather"))
lo, hi = (sys.argv[1] if len(sys.argv) > 1 else months[0]), (sys.argv[2] if len(sys.argv) > 2 else months[-1])
os.makedirs(OUT, exist_ok=True)
tmp = os.path.join(OUT, "_tmp")
os.makedirs(tmp, exist_ok=True)
con = duckdb.connect()
sql = T._BOOK30_SQL.format(table="bar1m")
for m in months:
    if not (lo <= m <= hi):
        continue
    if os.path.exists(os.path.join(OUT, f"{m}.parquet")) and "b_dep" in pd.read_parquet(os.path.join(OUT, f"{m}.parquet")).columns:
        continue
    d = pd.read_feather(os.path.join(F.DATA_1M, f"{m}.0.feather"), columns=["date", "instrument_id"] + COLS)
    d["instrument_id"] = d["instrument_id"].astype(int)
    for c in COLS:
        d[c] = d[c].astype("float64")
    inst = pd.DataFrame({"instrument_id": sorted(d["instrument_id"].unique())})
    inst["instrument"] = "ID" + inst["instrument_id"].astype(str).str.zfill(4)
    # duckdb 0.9 与 numpy 2 / pandas 3 的直接交换会崩溃：经 parquet 文件输入、fetchall 取回
    d.to_parquet(os.path.join(tmp, "bar1m.parquet"), index=False)
    inst.to_parquet(os.path.join(tmp, "inst.parquet"), index=False)
    con.execute(f"CREATE OR REPLACE VIEW bar1m AS SELECT * FROM read_parquet('{tmp}/bar1m.parquet')")
    con.execute(f"CREATE OR REPLACE VIEW all_instruments AS SELECT * FROM read_parquet('{tmp}/inst.parquet')")
    cur = con.execute(sql)
    w = pd.DataFrame(cur.fetchall(), columns=[c[0] for c in cur.description])
    w["date"] = pd.to_datetime(w["date"])
    for c in T.BOOK30_COLS + T.BOOK2_COLS + T.BOOK3_COLS:
        w[c] = w[c].astype("float32")
    w.sort_values(["date", "instrument"]).to_parquet(os.path.join(OUT, f"_{m}.parquet"), index=False)
    os.replace(os.path.join(OUT, f"_{m}.parquet"), os.path.join(OUT, f"{m}.parquet"))     # 原子替换，正在读缓存的训练不受影响
    print(m, len(d), "->", len(w), w[T.BOOK30_COLS + T.BOOK2_COLS + T.BOOK3_COLS].notna().mean().round(3).to_dict(), flush=True)
