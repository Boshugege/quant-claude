# -*- coding: utf-8 -*-
"""用合成面板数据执行 build_wave11 里真实的 pandas 片段，检查是否产出有效值。"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bw11", here / "build_wave11.py")
bw11 = importlib.util.module_from_spec(spec)
sys.modules["bw11"] = bw11
spec.loader.exec_module(bw11)

rng = np.random.default_rng(5)
insts = ["%06d.SZ" % i for i in range(40)]
days = pd.date_range("2023-01-02", periods=12, freq="B")
segs = ["09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00", "13:30:00", "14:00:00", "14:30:00"]
rows = [(pd.Timestamp("%s %s" % (d.date(), s)), i) for d in days for s in segs for i in insts]
df = pd.DataFrame(rows, columns=["date", "instrument"])
df["day"] = df["date"].dt.normalize()
df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
n = len(df)
df["n_min"] = rng.integers(20, 31, n)
df["sum_amt"] = rng.lognormal(15, 1.0, n)
df["sum_amt2"] = df["sum_amt"] ** 2 * rng.uniform(0.02, 0.1, n)
df["sum_amret"] = rng.gamma(2, 0.002, n)
df["sum_mret"] = (rng.random(n) - 0.5) * df["sum_amret"]
df["sum_mret2"] = rng.gamma(2, 1e-5, n)
df["sum_mret3"] = (rng.random(n) - 0.5) * df["sum_mret2"] * 0.05
df["up_n"] = rng.integers(0, 30, n)
df["dn_n"] = rng.integers(0, 30, n)
df["zr_n"] = np.maximum(0, df["n_min"] - df["up_n"] - df["dn_n"])
df["max_mret"] = rng.normal(0, 0.004, n)
df["sum_amt_aret"] = df["sum_amt"] * df["sum_amret"] * rng.uniform(0.2, 0.8, n)


def roll(s, k, fn, min_periods):
    return s.groupby(df["instrument"], sort=False).transform(
        lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())


ns = {"np": np, "pd": pd, "df": df, "roll": roll}
bad = []
for col, code in bw11.SNIPS.items():
    exec("if True:\n" + code, ns)
    v = ns["df"][col]
    ok = v.notna().sum()
    print("%-10s 非空 %4d / %d  标准差 %.4f" % (col, ok, len(v), float(np.nanstd(v.values))))
    if ok < len(v) * 0.3 or not np.isfinite(np.nanstd(v.values)) or np.nanstd(v.values) == 0:
        bad.append(col)
assert not bad, "无效片段：%s" % bad
print("PASS")
