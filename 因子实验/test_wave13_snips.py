# -*- coding: utf-8 -*-
"""执行 build_wave13 里真实的大单成本锚片段，检查是否产出有效值。"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("bw13", here / "build_wave13.py")
bw13 = importlib.util.module_from_spec(spec)
sys.modules["bw13"] = bw13
spec.loader.exec_module(bw13)

rng = np.random.default_rng(9)
insts = ["%06d.SZ" % i for i in range(20)]
days = pd.date_range("2023-01-02", periods=10, freq="B")
segs = ["09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00", "13:30:00", "14:00:00", "14:30:00"]
rows = [(pd.Timestamp("%s %s" % (d.date(), s)), i) for d in days for s in segs for i in insts]
df = pd.DataFrame(rows, columns=["date", "instrument"])
df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
n = len(df)
df["sum_amt"] = rng.lognormal(15, 1.0, n)
df["sum_amt2"] = df["sum_amt"] ** 2 * rng.uniform(0.02, 0.1, n)
px = pd.Series(rng.uniform(5, 50, n))
df["sum_amt2_px"] = df["sum_amt2"] * px
r_dvwap = pd.Series(rng.normal(0, 0.01, n))


def roll(s, k, fn, min_periods):
    return s.groupby(df["instrument"], sort=False).transform(
        lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())


def safe_log_ratio(num, den):
    ok = (num > 0) & (den > 0)
    return pd.Series(np.where(ok, np.log(num.where(ok, 1.0) / den.where(ok, 1.0)), np.nan), index=num.index)


ns = {"np": np, "pd": pd, "df": df, "roll": roll, "px": px, "r_dvwap": r_dvwap,
      "safe_log_ratio": safe_log_ratio}
for name, code, keys in [("附加式", bw13.SNIP_ADD_SCALE, ("r_bv4", "r_bv8")),
                         ("替换式", bw13.SNIP_REPLACE_SCALE, ("r_rv4", "r_rv8"))]:
    exec("if True:\n" + code, ns)
    for k in keys:
        v = ns["df"][k]
        print("%s %s 非空 %d/%d 标准差 %.4f" % (name, k, v.notna().sum(), len(v), float(np.nanstd(v.values))))
        assert v.notna().sum() > len(v) * 0.3 and np.nanstd(v.values) > 0
print("大单成本锚片段自测通过")
