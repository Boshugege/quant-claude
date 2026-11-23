# -*- coding: utf-8 -*-
"""用合成面板数据执行 build_wave3 里真实的 pandas 片段，检查是否产出有效值。"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
spec = importlib.util.spec_from_file_location("bw3", here / "build_wave3.py")
bw3 = importlib.util.module_from_spec(spec)
sys.modules["bw3"] = bw3
spec.loader.exec_module(bw3)

rng = np.random.default_rng(2)
insts = ["%06d.SZ" % i for i in range(40)]
days = pd.date_range("2023-01-02", periods=12, freq="B")
segs = ["09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00", "13:30:00", "14:00:00", "14:30:00"]
rows = [(pd.Timestamp("%s %s" % (d.date(), s)), i) for d in days for s in segs for i in insts]
df = pd.DataFrame(rows, columns=["date", "instrument"])
df["day"] = df["date"].dt.normalize()
n = len(df)
df["sum_amt"] = rng.lognormal(15, 1.0, n)
df["sum_mret2"] = rng.gamma(2, 1e-4, n)
df["win_cls"] = np.where(df["date"].dt.strftime("%H:%M:%S") == "14:30:00", rng.lognormal(14, 1.0, n), 0.0)


def roll(s, k, fn, min_periods):
    return s.groupby(df["instrument"], sort=False).transform(
        lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())


ns = {"np": np, "pd": pd, "df": df, "roll": roll}
ok = True
for col, code in bw3.SNIPS.items():
    exec("if True:\n" + bw3.PREV_DAYS + code, ns)
    s = ns["df"][col]
    print("%-11s 非空 %5d / %d  标准差 %.4f  分位[5,50,95]=%s"
          % (col, int(s.notna().sum()), len(s), float(s.std()),
             np.round(np.nanpercentile(s.astype(float), [5, 50, 95]), 4)))
    if s.notna().sum() == 0 or float(s.std()) == 0:
        ok = False
        print("   -> 警告：无有效值或方差为 0")
print("PASS" if ok else "FAIL")
