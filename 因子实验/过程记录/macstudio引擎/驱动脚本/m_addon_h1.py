# -*- coding: utf-8 -*-
"""10-27：1669 的 14 个附加项 + q（评分机五年任务的值）对齐到引擎面板，存 vc_tmp/h1_A_<名字>.npy（原始值，extra_series 逐截面排名），供 h2_robust --extra 进树。"""
import os, sys
import numpy as np
import pandas as pd
from c1_upgrades import G, OUT
from g_go import inst_codes
c = G.c
keys = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c)})
keys["_r"] = np.arange(len(keys))
SRC = os.path.expanduser("~/Desktop/work/quant/fx/addon")
for f in sorted(os.listdir(SRC)):
    n = f[:-8]
    d = pd.read_parquet(os.path.join(SRC, f))[["date", "instrument", "factor"]]
    d["date"] = pd.to_datetime(d["date"])
    d = d.drop_duplicates(["date", "instrument"], keep="last")
    m = keys.merge(d, on=["date", "instrument"], how="left")
    x = np.full(len(keys), np.nan, np.float32); x[m["_r"].to_numpy()] = m["factor"].to_numpy(np.float32)
    np.save(OUT / ("h1_A_%s.npy" % n), x)
    print(n, "覆盖 %.3f" % np.isfinite(x).mean(), flush=True)
