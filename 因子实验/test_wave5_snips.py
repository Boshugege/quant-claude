# -*- coding: utf-8 -*-
"""用合成面板数据执行 build_wave5 里真实的 pandas 片段，检查是否产出有效值。"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
spec = importlib.util.spec_from_file_location("bw5", here / "build_wave5.py")
bw5 = importlib.util.module_from_spec(spec)
sys.modules["bw5"] = bw5
spec.loader.exec_module(bw5)

rng = np.random.default_rng(3)
insts = ["%06d.SZ" % i for i in range(40)]
days = pd.date_range("2023-01-02", periods=12, freq="B")
segs = ["09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00", "13:30:00", "14:00:00", "14:30:00"]
rows = [(pd.Timestamp("%s %s" % (d.date(), s)), i) for d in days for s in segs for i in insts]
df = pd.DataFrame(rows, columns=["date", "instrument"])
df["day"] = df["date"].dt.normalize()
df = df.sort_values(["instrument", "date"]).reset_index(drop=True)
n = len(df)
df["sum_amt"] = rng.lognormal(15, 1.0, n)
df["r_30"] = rng.normal(0, 0.01, n)
df["mcap"] = rng.lognormal(22, 1.0, n)


def roll(s, k, fn, min_periods):
    return s.groupby(df["instrument"], sort=False).transform(
        lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())


ns = {"np": np, "pd": pd, "df": df, "roll": roll}
ok = True
for col, code in [("c_turn", bw5.SNIP_TURN), ("c_mcap2", bw5.SNIP_MCAP2),
                  ("c_m30", bw5.SNIP_M30), ("c_midx", bw5.SNIP_MIDX)]:
    exec("if True:\n" + code, ns)
    s = ns["df"][col]
    print("%-9s 非空 %5d / %d  标准差 %.4f  分位[5,50,95]=%s"
          % (col, int(s.notna().sum()), len(s), float(s.std()),
             np.round(np.nanpercentile(s.astype(float), [5, 50, 95]), 4)))
    if s.notna().sum() == 0 or float(s.std()) == 0:
        ok = False
        print("   -> 警告：无有效值或方差为 0")
print("PASS" if ok else "FAIL")
