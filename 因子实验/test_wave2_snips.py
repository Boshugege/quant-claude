# -*- coding: utf-8 -*-
"""用合成面板数据执行 build_wave2 里真实的 pandas 片段，检查是否产出有效值。"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

here = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(here))
spec = importlib.util.spec_from_file_location("bw2", here / "build_wave2.py")
bw2 = importlib.util.module_from_spec(spec)
sys.modules["bw2"] = bw2
spec.loader.exec_module(bw2)

rng = np.random.default_rng(1)
insts = ["%06d.SZ" % i for i in range(40)]
days = pd.date_range("2023-01-02", periods=12, freq="B")
segs = ["09:30:00", "10:00:00", "10:30:00", "11:00:00", "11:30:00", "13:30:00", "14:00:00", "14:30:00"]
rows = [(pd.Timestamp("%s %s" % (d.date(), s)), i) for d in days for s in segs for i in insts]
df = pd.DataFrame(rows, columns=["date", "instrument"])
df["day"] = df["date"].dt.normalize()
n = len(df)
df["sum_amt"] = rng.lognormal(15, 1.0, n)
df["sum_samt"] = df["sum_amt"] * rng.uniform(-0.3, 0.3, n)
df["win_auc"] = np.where(df["date"].dt.strftime("%H:%M:%S") == "09:30:00", rng.lognormal(14, 1.0, n), 0.0)


def roll(s, k, fn, min_periods):
    return s.groupby(df["instrument"], sort=False).transform(
        lambda x: getattr(x.rolling(k, min_periods=min_periods), fn)())


ns = {"np": np, "pd": pd, "df": df, "roll": roll}
ok = True
for cols in [("c_auс"[:0] + "c_auc",), ("c_aucshare",), ("c_netflow",)]:
    body = bw2.PREV_DAYS if any(c in ("c_auc", "c_aucshare") for c in cols) else ""
    for c in cols:
        body += {"c_auc": bw2.SNIP_AUC, "c_aucshare": bw2.SNIP_AUCSHARE, "c_netflow": bw2.SNIP_NETFLOW}[c]
    exec("if True:\n" + body, ns)
    for c in cols:
        s = ns["df"][c]
        print("%-12s 非空 %5d / %d  标准差 %.4f  分位[5,50,95]=%s"
              % (c, int(s.notna().sum()), len(s), float(s.std()),
                 np.round(np.nanpercentile(s.astype(float), [5, 50, 95]), 4)))
        if s.notna().sum() < len(s) * 0.5 or float(s.std()) == 0:
            ok = False
            print("   -> 警告：有效值/方差异常")
print("PASS" if ok else "FAIL")
