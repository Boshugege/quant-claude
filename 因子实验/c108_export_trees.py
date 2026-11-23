# -*- coding: utf-8 -*-
"""T11（10-22）：把树的多模型平均（h2_<tag>_<年>.npy）导出为 2023–24 的因子值（date／instrument／factor），
送 personal 评分机（kind=values）做底座，再加规模比 + 缓冲带比较。用法：c108_export_trees.py <名字> <标签1+标签2+…>"""
import sys
import numpy as np
import pandas as pd
from c1_upgrades import G, OUT
from g_go import inst_codes
c = G.c; z = G.z930
kd = pd.to_datetime(c.key_date); ins = inst_codes(c)
name, tags = sys.argv[1], sys.argv[2].split("+")
parts = []
for yr in ("2023", "2024"):
    x = np.mean([np.load(OUT / ("h2_%s_%s.npy" % (t, yr))) for t in tags], axis=0)
    x = np.where(z, 0.0, x)
    m = np.asarray(kd.year == int(yr))
    parts.append(pd.DataFrame({"date": kd[m], "instrument": ins[np.flatnonzero(m)], "factor": x[m]}))
o = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "instrument"], keep="last")
o.to_parquet(OUT / ("val_%s_2324.parquet" % name), index=False)
print("导出", name, len(o), flush=True)
