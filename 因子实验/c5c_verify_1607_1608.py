# -*- coding: utf-8 -*-
"""1607、1608 代码版（评测机 2024 不分块任务的 factor_raw）逐行对照参考：
  1607 参考 = h2_robust 导出的 h2_c2pf_2024.npy（模型 h2_c2_pf.json 在 engine 面板上的预测）；
  1608 参考 = 1363（含它内部的九根，代码里本地照算）+ 六个安全根再加一倍 + CLOSEPOS20 0.0865。
用法（personal）：PYTHONPATH=因子实验 ~/jyc_scorer/.venv/bin/python 因子实验/c5c_verify_1607_1608.py"""
import numpy as np
import pandas as pd
from c1_upgrades import G, CP, ax, W1363, DEAL, MET, OUT
from g_go import inst_codes

c = G.c; z930 = G.z930
kd = pd.to_datetime(c.key_date)
m = (kd.year == 2024)
ic = inst_codes(c)[np.flatnonzero(m)]


def check(name, B, job):
    B = np.asarray(B, float).copy(); B[z930] = 0.0
    ref = pd.DataFrame({"date": kd[m], "instrument": ic, "ref": B[m]})
    a = pd.read_parquet("/home/user/jyc_scorer/jobs/%s/factor_raw.parquet" % job)
    a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
    x = ref.merge(a, on=["date", "instrument"])
    x = x[x["date"].dt.strftime("%H%M") != "0930"]
    rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
    d = (x["factor"] - x["ref"]).abs()
    print("%s %s 逐截面秩相关 均值 %.5f 中位 %.5f 最小 %.5f；逐行 |差| 中位 %.5f p99 %.5f；行 %d（代码 %d）" % (
        name, job, rc.mean(), rc.median(), rc.min(), d.median(), d.quantile(0.99), len(x), len(a)), flush=True)
    mr = G.ev(B, "2024")
    print("  参考 2024 五项 %s" % " / ".join("%.4f" % mr[k] for k in MET), flush=True)


h2 = np.load(OUT / "h2_c2pf_2024.npy")
assert len(h2) == c.n_rows, (len(h2), c.n_rows)
check("1607", h2, "1018-011751-6b53")
B = np.asarray(G.B[("1363", 0)], float) + sum(W1363[r] * ax(r, 0) for r in W1363 if r not in DEAL) + 0.0865 * CP
check("1608", B, "1018-020410-694f")
