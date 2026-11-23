# -*- coding: utf-8 -*-
"""在本地 1 分钟数据上用 DuckDB（0.9.2，与平台同代）原样执行提交脚本的 _WIN30_SQL，
与本地训练用的 30 分钟窗口表（feather_data.windows30）逐行比较；同时对比修复前的旧写法。
用法：bash e2e/scripts/remote.sh e2e/scripts/sql_win30_check.py [YYYY-MM]
"""
import os
import sys

import duckdb
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import feather_data as F  # noqa: E402
import Transformer_modelsave_train as T  # noqa: E402

month = sys.argv[1] if len(sys.argv) > 1 else "2024-01"
F.set_read_cols(["open", "high", "low", "close", "volume", "amount", "deal_number", "bid_price1", "ask_price1", "adjust_factor"])
d = F.read_month(month.replace("-", ""))
d = d.sort_values(["instrument", "date"]).reset_index(drop=True)
d["day"] = d["date"].dt.normalize()
last = d.groupby(["instrument", "day"]).agg(dc=("close", "last"), da=("adjust_factor", "last"))
g = last.groupby(level=0)
last["pre_close"] = g["dc"].shift(1) * g["da"].shift(1) / last["da"]
d = d.merge(last[["pre_close"]].reset_index(), on=["instrument", "day"], how="left")
d["instrument_id"] = d["instrument"].str[2:].astype(int)
bar1m = d[["date", "instrument_id", "bid_price1", "ask_price1", "open", "high", "low", "close", "pre_close", "volume", "amount", "deal_number"]]
all_instruments = d[["instrument_id", "instrument"]].drop_duplicates()
# duckdb 0.9 与 numpy 2 / pandas 3 的直接交换会崩溃：经 parquet 文件输入、fetchall 取回
tmp = os.path.join("jobs", "sql_win30_tmp")
os.makedirs(tmp, exist_ok=True)
bar1m.to_parquet(os.path.join(tmp, "bar1m.parquet"), index=False)
all_instruments.to_parquet(os.path.join(tmp, "inst.parquet"), index=False)
con = duckdb.connect()
con.execute(f"CREATE VIEW bar1m AS SELECT * FROM read_parquet('{tmp}/bar1m.parquet')")
con.execute(f"CREATE VIEW all_instruments AS SELECT * FROM read_parquet('{tmp}/inst.parquet')")
new_sql = T._WIN30_SQL.format(table="bar1m")
old_sql = new_sql.replace(" OR (time_segment = 93000 AND hms = '092500')", "")
assert old_sql != new_sql
ref = F.windows30(f"{month}-01", (pd.Timestamp(f"{month}-01") + pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d 23:59:59"),
                  all_instruments["instrument"].tolist())
for name, sql in [("修复后", new_sql), ("修复前", old_sql)]:
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    w = pd.DataFrame(cur.fetchall(), columns=cols)
    w["date"] = pd.to_datetime(w["date"])
    w["last_px"] = w["last_px"].where(w["last_px"] > 0)
    w = w.merge(ref, on=["date", "instrument"], suffixes=("", "_ref"))
    w["hm"] = w["date"].dt.strftime("%H:%M")
    ok_px = ((w.last_px - w.last_px_ref).abs() < 1e-4) | (w.last_px.isna() & w.last_px_ref.isna())
    ok_amt = (w.sum_amt - w.sum_amt_ref).abs() <= 1e-6 * w.sum_amt_ref.abs().clip(lower=1)
    print(f"== {name}：行数 {len(w):,}")
    print("  末价一致率", ok_px.groupby(w.hm).mean().round(4).to_dict())
    print("  末价缺失率", w.last_px.isna().groupby(w.hm).mean().round(4).to_dict())
    print("  成交额一致率", ok_amt.groupby(w.hm).mean().round(4).to_dict())
    if name == "修复后":                                    # E07 S4 起新增的窗口开高低收、原始成交额、成交笔数
        for c in ("w_open", "w_high", "w_low", "w_close", "raw_amt", "w_deals"):
            a, b = w[c].astype(float), w[c + "_ref"].astype(float)
            ok = ((a - b).abs() <= 1e-4 * b.abs().clip(lower=1)) | (a.isna() & b.isna())
            print(f"  {c} 一致率", round(float(ok.mean()), 4), ok.groupby(w.hm).mean().round(3).to_dict())
            if c == "w_open":
                bad = w[~ok & (w.hm == ok.groupby(w.hm).mean().idxmin())]
                print(bad[["date", "instrument", "w_open", "w_open_ref", "w_high", "w_high_ref", "w_low", "w_low_ref"]].head(5).to_string())
