# 10-24：检查新辅助目标 au（收盘竞价跳动）／dr（连续竞价漂移）只在 14:30 有值，且两者之和与官方标签高度相关。
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend
fb = FeatherBackend()
lo, hi = "2023-03-01 00:00:00", "2023-03-31 23:59:59"
inst = fb.pool(lo, hi)[:300]
sec = fb.sections(lo, hi, inst)
lab = fb.labels(lo, hi, inst)
sec = sec.merge(lab[["date", "instrument", "y"]], on=["date", "instrument"], how="left")
bars = fb.bars(lo, hi, sorted(sec["instrument"].unique().tolist()), T.FEATURE_COLS + ["adjust_factor"])
keys = sec[["date", "instrument", "y"]].copy()
a = T._future_aux(bars, keys)
hm = keys["date"].dt.strftime("%H%M")
for c in ("au", "dr"):
    v = pd.Series(a[c])
    print(c, "覆盖：14:30 %.3f，其它 %.3f" % (v[hm == "1430"].notna().mean(), v[hm != "1430"].notna().mean()))
k = keys[hm == "1430"].assign(au=a["au"][hm == "1430"], dr=a["dr"][hm == "1430"])
k["s"] = k["au"] + k["dr"]
ic = lambda c: k.groupby("date")[[c, "y"]].apply(lambda g: g[c].corr(g["y"], method="spearman")).mean()
print("14:30 与官方标签的截面秩相关：au %.3f  dr %.3f  au+dr %.3f" % (ic("au"), ic("dr"), ic("s")))
