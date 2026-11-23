# -*- coding: utf-8 -*-
"""本地因子库与平台因子值（平台探测）的逐日截面秩相关：判断差异来自预热（随时间上升）还是定义（恒定）。
用法：bash 工具/本地评分/remote.sh e2e/scripts/fac_impl_diff.py
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "工具/本地评分")
import localeval as le
from factor_lib import FACTORS, add_components
p = add_components(le.build_panel(le.load_bars("2023-11", "2024-03")))
hh = (p["slot"] // 100).astype(int).astype(str).str.zfill(2) + ":" + (p["slot"] % 100).astype(int).astype(str).str.zfill(2)
p["key_date"] = p["day"] + pd.to_timedelta(hh + ":00")
mp = pd.read_csv("数据/exposure/id_map.csv"); mp = mp[mp.ok]; mp["instrument_id"] = mp.local_id.str[2:].astype(int)
facs = {f[0]: f for f in FACTORS}
out = {}
for fid in ["90", "98", "100", "102", "104"]:
    a = facs[fid][3](p)
    pf = pd.read_parquet(f"数据/平台探测/probe_fac_{fid}.parquet").merge(mp[["instrument", "instrument_id"]], on="instrument").rename(columns={"date": "key_date"})
    d = p[["key_date", "instrument_id", "day", "slot"]].assign(a=a.values).merge(pf[["key_date", "instrument_id", "factor"]], on=["key_date", "instrument_id"])
    rc = d.groupby("key_date").apply(lambda g: g["a"].rank().corr(g["factor"].rank()))
    wk = rc.groupby(rc.index.to_period("W")).mean()
    out[fid] = wk
    sl = d.assign(rc=d.key_date.map(rc)).groupby("slot")["rc"].mean()
    print(fid, "按时段", sl.round(3).to_dict())
print(pd.DataFrame(out).round(3).to_string())
