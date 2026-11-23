# -*- coding: utf-8 -*-
"""用合成面板数据执行 build_micro.SNIPS 里的真实片段，检查是否产生有效值（无静默全 NaN）。"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

p = pathlib.Path(__file__).resolve().parent / "build_micro.py"
spec = importlib.util.spec_from_file_location("bm", p)
bm = importlib.util.module_from_spec(spec)
sys.modules["bm"] = bm
spec.loader.exec_module(bm)

rng = np.random.default_rng(0)
insts = ["%06d.SZ" % i for i in range(40)]
days = pd.date_range("2023-01-02", periods=12, freq="B")
segs = ["09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00", "13:30:00", "14:00:00", "14:30:00"]
rows = []
for d in days:
    for s in segs:
        for i in insts:
            rows.append((pd.Timestamp("%s %s" % (d.date(), s)), i))
df = pd.DataFrame(rows, columns=["date", "instrument"])
df["day"] = df["date"].dt.normalize()
n = len(df)
df["sum_amt"] = rng.lognormal(15, 1.0, n)
df["sum_vol"] = df["sum_amt"] / rng.uniform(8, 20, n)
df["sum_amt2"] = df["sum_amt"] ** 2 * rng.uniform(0.2, 0.5, n)
df["max_min_amt"] = df["sum_amt"] * rng.uniform(0.1, 0.5, n)
df["sum_amt_mn"] = df["sum_amt"] * (df["date"].dt.hour * 60 + df["date"].dt.minute
                                       + rng.uniform(-12, 12, n))
df["sum_mret2"] = rng.gamma(2, 1e-4, n)
df["sum_mret2_dn"] = df["sum_mret2"] * rng.uniform(0.2, 0.8, n)
df["last_px"] = rng.uniform(5, 50, n)
df["pre_close"] = df["last_px"] * rng.uniform(0.9, 1.1, n)
df["r_30"] = rng.normal(0, 0.01, n)


def roll(s, k, fn, min_periods):
    return s.groupby(df["instrument"], sort=False).transform(
        lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())


ns = {"np": np, "pd": pd, "df": df, "roll": roll}
ok = True
for col, code in bm.SNIPS.items():
    try:
        exec("if True:\n" + code, ns)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("片段执行失败", col, type(exc).__name__, exc)
        continue
    s = ns["df"][col]
    print("%-10s 非空 %6d / %d  标准差 %.4f  分位[5,50,95]=%s"
          % (col, int(s.notna().sum()), len(s), float(s.std()),
             np.round(np.nanpercentile(s.astype(float), [5, 50, 95]), 4)))
    if s.notna().sum() < len(s) * 0.5:
        ok = False
        print("   -> 警告：有效值过少")
print("PASS" if ok else "FAIL")
