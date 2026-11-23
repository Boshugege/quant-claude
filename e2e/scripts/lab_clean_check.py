# -*- coding: utf-8 -*-
"""官方标签的 VWAP 是否做了分钟级清洗：用 2021-06（异常成交最多）比较官方 vwap 与本地 1 分钟
“不清洗 / 清洗（[low×0.98, high×1.02]）”两种 VWAP，重点看含异常分钟的窗口。
用法：bash e2e/scripts/remote.sh e2e/scripts/lab_clean_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "e2e/src")
import feather_data as F  # noqa: E402

mp = pd.read_csv("数据/exposure/id_map.csv")
mp = mp[mp.ok]
lab = pd.read_parquet("数据/平台探测/probe_label_202106.parquet").merge(mp[["instrument", "local_id"]], on="instrument")
F.set_read_cols(["high", "low", "close", "volume", "amount"])
b = F.read_range("2021-06-01 00:00:00", "2021-06-30 23:59:59", sorted(lab.local_id.unique()))
hm = (b.date.dt.hour * 100 + b.date.dt.minute).to_numpy()
k = np.searchsorted(F._BUCKET_HHMM, hm, side="left")
b = b.assign(day=b.date.dt.normalize(), w=np.minimum(k, 8))
mv = b.amount / b.volume.where(b.volume > 0)
b["bad"] = b.volume.gt(0) & ~((mv >= b.low * 0.98) & (mv <= b.high * 1.02))
b["amt_c"] = b.amount.where(~b.bad, 0.0)
b["vol_c"] = b.volume.where(~b.bad, 0.0)
g = b.groupby(["instrument", "day", "w"]).agg(a=("amount", "sum"), v=("volume", "sum"), ac=("amt_c", "sum"), vc=("vol_c", "sum"),
                                             nbad=("bad", "sum")).reset_index()
SL = [930, 1000, 1030, 1100, 1130, 1330, 1400, 1430]
g = g[g.w >= 1]
g["sec"] = [SL[i - 1] for i in g.w]
lab["sec"] = lab.date.dt.hour * 100 + lab.date.dt.minute
lab["day"] = lab.date.dt.normalize()
m = lab.merge(g.rename(columns={"instrument": "local_id"}), on=["local_id", "day", "sec"])
m["v_raw"] = m.a / m.v
m["v_cln"] = m.ac / m.vc.where(m.vc > 0)
bad = m[m.nbad > 0]
print("含异常分钟的窗口数", len(bad), "（总窗口", len(m), "）；按时段", bad.groupby("sec").size().to_dict())
for c in ["v_raw", "v_cln"]:
    e = ((bad[c] - bad.vwap) / bad.vwap).abs()
    print(f"  {c}: 与官方 vwap 相对差 <1e-4 的比例 {float((e < 1e-4).mean()):.3f}，中位相对差 {float(e.median()):.2e}")
print(bad[["date", "instrument", "vwap", "v_raw", "v_cln", "end_price", "nbad"]].head(8).to_string())
