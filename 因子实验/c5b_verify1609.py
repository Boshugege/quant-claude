# -*- coding: utf-8 -*-
"""1609 代码版（c1609 = 评测机 2024 不分块任务的 factor_raw）对照 engine 重建的参考（= c11_dose1609 的剂量 1.0），逐截面秩相关与逐行差。
用法（personal）：PYTHONPATH=因子实验 ~/jyc_scorer/.venv/bin/python 因子实验/c5b_verify1609.py 1018-050535-70a6"""
import sys
import numpy as np
import pandas as pd
import root_eval as R
from c1_upgrades import G, CP, CLS, ax, W1363, SMALL, DEAL, MET
from g_go import inst_codes

job = sys.argv[1]
c = G.c; z930 = G.z930; slot = c.slot_of_date[c.date_id]
K = 0.2908 / 0.289
B = np.asarray(R.load("fm_s152")[0], float).copy(); B[z930] = 0.0
for r in [r for r in W1363 if r not in DEAL]:
    B += W1363[r] * ax(r, 0)
for k, w in SMALL.items():
    B += w * G.axis(k, 0)
B += 0.10 * K * CP + 0.05 * K * CLS + 0.10 * K * CLS * (slot == 1430)
B[z930] = 0.0
kd = pd.to_datetime(c.key_date)
m = (kd.year == 2024)
ref = pd.DataFrame({"date": kd[m], "instrument": inst_codes(c)[np.flatnonzero(m)], "ref": B[m]})
a = pd.read_parquet("/home/user/jyc_scorer/jobs/%s/factor_raw.parquet" % job)
a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
x = ref.merge(a, on=["date", "instrument"])
x = x[x["date"].dt.strftime("%H%M") != "0930"]
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
d = (x["factor"] - x["ref"]).abs()
print("1609 %s 逐截面秩相关 均值 %.5f 中位 %.5f 最小 %.5f；逐行 |差| 中位 %.5f p99 %.5f；行 %d（参考 %d、代码 %d）" % (
    job, rc.mean(), rc.median(), rc.min(), d.median(), d.quantile(0.99), len(x), int(m.sum()), len(a)), flush=True)
mr = G.ev(B, "2024")
print("参考 2024 五项 %s" % " / ".join("%.4f" % mr[k] for k in MET), flush=True)
