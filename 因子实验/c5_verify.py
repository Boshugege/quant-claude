# -*- coding: utf-8 -*-
"""Claude 夜间：代码版（评测机 2024 不分块任务的 factor_raw）对照"代码等价"的 values 参考（engine 口径重建），
看逐截面秩相关与 2024 五项。用法：python c5_verify.py <种类 1603|1601|1602s152> <任务号>
代码等价 = 平台一致版 + 1363 内部仍含的三个成交笔数类根（本地算得出来）。"""
import sys
import pathlib

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c1_upgrades import G, build, ax, W1363, DEAL, MET  # noqa: E402
from g_go import inst_codes  # noqa: E402

kind, job = sys.argv[1], sys.argv[2]
c = G.c
if kind in ("1603", "1601"):
    B = build(kind, 0, dose=1.5, cp=0.05) + sum(W1363[r] * G.axis(r, 0) for r in DEAL)
else:
    import c1b_heads as H                                                 # noqa: E402
    B = H.build("s152", 0, cp=0.05)
B[G.z930] = 0.0
kd = pd.to_datetime(c.key_date)
m = (kd.year == 2024)
ref = pd.DataFrame({"date": kd[m], "instrument": inst_codes(c)[np.flatnonzero(m)], "ref": B[m]})
a = pd.read_parquet("/home/user/jyc_scorer/jobs/%s/factor_raw.parquet" % job)
a["date"] = pd.to_datetime(a["date"])
a["instrument"] = a["instrument"].astype(str)
x = ref.merge(a, on=["date", "instrument"])
x = x[x["date"].dt.strftime("%H%M") != "0930"]
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
print("%s %s 逐截面秩相关 均值 %.5f 最小 %.5f 行 %d" % (kind, job, rc.mean(), rc.min(), len(x)), flush=True)
mr = G.ev(B, "2024")
print("参考 2024 五项 %s" % " / ".join("%.4f" % mr[k] for k in MET), flush=True)
