# -*- coding: utf-8 -*-
"""通用片段自测：python3 因子实验/test_snips_generic.py <模块名> <SNIPS 属性名>

用合成面板数据（含 wave1–wave6 用到的全部列）执行真实的 pandas 片段，检查是否产出有效值。
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
mod_name, attr = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(mod_name, here / (mod_name + ".py"))
mod = importlib.util.module_from_spec(spec)
sys.modules[mod_name] = mod
spec.loader.exec_module(mod)
snips = getattr(mod, attr)

rng = np.random.default_rng(7)
insts = ["%06d.SZ" % i for i in range(40)]
days = pd.date_range("2023-01-02", periods=12, freq="B")
segs = ["09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00", "13:30:00", "14:00:00", "14:30:00"]
rows = [(pd.Timestamp("%s %s" % (d.date(), s)), i) for d in days for s in segs for i in insts]
df = pd.DataFrame(rows, columns=["date", "instrument"])
df["day"] = df["date"].dt.normalize()
df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
n = len(df)
df["sum_amt"] = rng.lognormal(15, 1.0, n)
df["sum_vol"] = df["sum_amt"] / 10
df["sum_samt"] = df["sum_amt"] * rng.uniform(-0.3, 0.3, n)
df["sum_amt2"] = df["sum_amt"] ** 2 * rng.uniform(0.2, 0.5, n)
df["max_min_amt"] = df["sum_amt"] * rng.uniform(0.1, 0.5, n)
df["sum_amt_mn"] = df["sum_amt"] * (df["date"].dt.hour * 60 + df["date"].dt.minute + rng.uniform(-12, 12, n))
df["sum_mret2"] = rng.gamma(2, 1e-4, n)
df["sum_mret2_dn"] = df["sum_mret2"] * rng.uniform(0.2, 0.8, n)
df["sum_rng"] = rng.gamma(2, 1e-3, n)
df["sum_upsh"] = df["sum_rng"] * rng.uniform(0.2, 0.6, n)
df["sum_dnsh"] = df["sum_rng"] * rng.uniform(0.2, 0.6, n)
df["sum_body"] = df["sum_rng"] * rng.uniform(0.1, 0.5, n)
df["win_high"] = rng.uniform(10, 50, n)
df["win_low"] = df["win_high"] * rng.uniform(0.95, 0.99, n)
df["win_cls"] = np.where(df["date"].dt.strftime("%H:%M:%S") == "14:30:00", rng.lognormal(14, 1, n), 0.0)
df["win_auc"] = np.where(df["date"].dt.strftime("%H:%M:%S") == "09:30:00", rng.lognormal(14, 1, n), 0.0)
df["last_px"] = rng.uniform(5, 50, n)
df["pre_close"] = df["last_px"] * rng.uniform(0.9, 1.1, n)
df["mcap"] = rng.lognormal(22, 1.0, n)
df["r_30"] = rng.normal(0, 0.01, n)


def roll(s, k, fn, min_periods):
    return s.groupby(df["instrument"], sort=False).transform(
        lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())


ns = {"np": np, "pd": pd, "df": df, "roll": roll}
if hasattr(mod, "PREV_DAYS"):
    exec("if True:\n" + mod.PREV_DAYS, ns)
ok = True
for col, code in snips.items():
    try:
        exec("if True:\n" + code, ns)
    except Exception as exc:  # noqa: BLE001
        ok = False
        print("FAIL %-10s %s: %s" % (col, type(exc).__name__, exc))
        continue
    s = ns["df"][col]
    std = float(s.std())
    print("%-11s 非空 %5d / %d  标准差 %.4f  分位[5,50,95]=%s"
          % (col, int(s.notna().sum()), len(s), std,
             np.round(np.nanpercentile(s.astype(float), [5, 50, 95]), 4)))
    if s.notna().sum() == 0 or std == 0:
        ok = False
        print("   -> 警告：无有效值或方差为 0")
print("PASS" if ok else "FAIL")
