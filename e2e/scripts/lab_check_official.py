# -*- coding: utf-8 -*-
"""本地标签（localeval 的 y_off）与官方 vwap_return（平台探测导出的 cpt_jyc_2026_vwap，2024Q1）逐行对比。
用法：bash 工具/本地评分/remote.sh e2e/scripts/lab_check_official.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "工具/本地评分")
import localeval as le  # noqa: E402

p = le.build_panel(le.load_bars("2024-01", "2024-03"))
p = p[p.is_eval]
hh = (p["slot"] // 100).astype(int).astype(str).str.zfill(2) + ":" + (p["slot"] % 100).astype(int).astype(str).str.zfill(2)
p["date"] = p["day"] + pd.to_timedelta(hh + ":00")
mp = pd.read_csv("数据/exposure/id_map.csv")
mp = mp[mp.ok]
mp["instrument_id"] = mp.local_id.str[2:].astype(int)
lab = pd.read_parquet("数据/平台探测/probe_label_2024Q1.parquet").merge(mp[["instrument", "instrument_id"]], on="instrument")
m = lab.merge(p[["date", "instrument_id", "slot", "y_off"]], on=["date", "instrument_id"], how="inner")
y = m.vwap_return.astype(float)
print("匹配行数", len(m))
for c in ["y_off"]:
    ok = np.isfinite(m[c]) & np.isfinite(y)
    rc = m[ok].groupby("date").apply(lambda g: g[c].rank().corr(g.vwap_return.rank()))
    print(c, "覆盖", round(ok.mean(), 4), "精确一致(<1e-5)", round(float(((m[c] - y).abs() < 1e-5)[ok].mean()), 4),
          "截面秩相关均值", round(rc.mean(), 4))
    print("   各时段覆盖", ok.groupby(m.slot).mean().round(3).to_dict())

# 差异诊断：按时段的精确一致率与秩相关；不一致行的 close / vwap 各自差多少
ok = np.isfinite(m["y_off"]) & np.isfinite(y)
mm = m[ok].copy()
mm["ex"] = (mm["y_off"] - mm["vwap_return"]).abs() < 1e-5
print("各时段精确一致率", mm.groupby("slot")["ex"].mean().round(3).to_dict())
print("各时段秩相关", mm.groupby("slot").apply(lambda g: g.groupby("date").apply(lambda h: h["y_off"].rank().corr(h["vwap_return"].rank())).mean()).round(3).to_dict())
q = p[["date", "instrument_id"]].copy()
g = p.groupby("instrument_id")
q["nc"] = g["close"].shift(-1).values
q["nv"] = g["vwap_w"].shift(-1).values
q["nvr"] = (g["amount"].shift(-1) / g["volume"].shift(-1)).values
mm = mm.merge(q, on=["date", "instrument_id"], how="left")
bad = mm[~mm["ex"]]
print("不一致行：收盘价相同比例", round(float(((bad.nc - bad.end_price).abs() < 1e-4).mean()), 3),
      " VWAP 相对差中位数", float(((bad.nv - bad.vwap) / bad.vwap).abs().median()),
      " 未清洗 VWAP 相对差中位数", float(((bad.nvr - bad.vwap) / bad.vwap).abs().median()))
print(bad[["date", "instrument", "vwap_return", "y_off", "end_price", "nc", "vwap", "nv"]].head(8).to_string())
