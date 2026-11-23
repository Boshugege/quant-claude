# -*- coding: utf-8 -*-
"""1611 代码版逐行核对：参考 = 1379 全含（本地成交笔数类根都能算：1363 + 0.5×九根）+ CLS20 0.0865（14:30 +0.173）+ CP 0.1298。用法：c5f_verify1611.py <任务号>"""
import sys
import numpy as np
import pandas as pd
from c1_upgrades import G, CP, CLS, W1363, MET
from g_go import inst_codes
job = sys.argv[1]
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
B = np.asarray(G.B[("1363", 0)], float) + sum(0.5 * W1363[r] * G.axis(r, 0) for r in W1363) + 0.0865 * CLS + 0.173 * CLS * (slot == 1430) + 0.1298 * CP
B[z] = 0.0
kd = pd.to_datetime(c.key_date); m = (kd.year == 2024)
ref = pd.DataFrame({"date": kd[m], "instrument": inst_codes(c)[np.flatnonzero(m)], "ref": B[m]})
a = pd.read_parquet("/home/user/jyc_scorer/jobs/%s/factor_raw.parquet" % job)
a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
x = ref.merge(a, on=["date", "instrument"]); x = x[x["date"].dt.strftime("%H%M") != "0930"]
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
d = (x["factor"] - x["ref"]).abs()
print("1611 %s 逐截面秩相关 均值 %.5f 中位 %.5f 最小 %.5f；逐行 |差| 中位 %.5f p99 %.5f；行 %d" % (job, rc.mean(), rc.median(), rc.min(), d.median(), d.quantile(0.99), len(x)), flush=True)
mr = G.ev(B, "2024")
print("参考 2024 五项 %s" % " / ".join("%.4f" % mr[k] for k in MET), flush=True)
