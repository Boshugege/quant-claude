# -*- coding: utf-8 -*-
"""回归测试：训练 [val] 指标（src/metrics.py）与本地评分器（localeval.evaluate）对同一份预测应给出一致结果。
用法：bash e2e/scripts/remote.sh e2e/scripts/metrics_consistency.py <pred.parquet> <YYYY>
"""
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "工具/本地评分")
import localeval as le
from metrics import evaluate
from feather_data import FeatherBackend
src, year = sys.argv[1], sys.argv[2]
pred = pd.read_parquet(src)[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)
p = le.build_panel(le.load_bars(f"{year}-01", f"{year}-12"))
m = p[["key_date", "instrument"]].merge(pred.rename(columns={"date": "key_date"}), on=["key_date", "instrument"], how="left")
a = le.evaluate(pd.Series(m["factor"].to_numpy(), index=p.index), p)
fb = FeatherBackend()
lab = fb.labels(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59", sorted(pred.instrument.unique()))
k = pred.merge(lab[["date", "instrument", "y"]], on=["date", "instrument"], how="left")
ex = fb.exposure(f"{year}-01-01", f"{year}-12-31", sorted(pred.instrument.unique()))
kd = k[["date", "instrument"]].assign(_day=k["date"].dt.normalize()).merge(ex.rename(columns={"date": "_day"}), on=["_day", "instrument"], how="left")
b = evaluate(k, k["factor"].to_numpy(), k["y"].to_numpy(), style=kd[le.NEUTRAL_COLS].to_numpy(np.float64))
print("localeval:", {x: round(float(a[x]), 4) for x in ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]}, "截面", a["n_xsec"])
print("metrics  :", {"ic_mean": round(b["ic"], 4), "ic_ir": round(b["icir"], 4), "ls_sharpe": round(b["ls_sharpe"], 4), "stress": round(b["stress"], 4), "turnover": round(b["turnover"], 4)}, "截面", b["sections"])
