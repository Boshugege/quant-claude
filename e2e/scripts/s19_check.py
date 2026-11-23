# S19 核对：用 _labels_from_bars 在官方截面上重算标签，与本地标签缓存（官方口径）逐时刻比较；再看中间时刻标签的覆盖率与分布。
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend

fb = FeatherBackend()
lo, hi = "2023-03-01 00:00:00", "2023-03-31 23:59:59"
insts = fb.pool(lo, hi)
sec = fb.sections(lo, hi, insts)
bars = fb.bars(pd.Timestamp(lo) - pd.Timedelta(days=7), hi, insts, T.FEATURE_COLS)
mine = T._labels_from_bars(bars, sec)
ref = fb.labels(lo, hi, insts)
d = mine.merge(ref, on=["date", "instrument"], suffixes=("_m", "_r")).dropna()
d["hm"] = d["date"].dt.strftime("%H:%M")
print("官方截面：重算 vs 缓存（逐时刻截面秩相关均值 / 最大绝对差）")
for hm, g in d.groupby("hm"):
    rc = g.groupby("date")[["y_m", "y_r"]].corr(method="spearman").xs("y_m", level=1)["y_r"].mean()
    print(f"  {hm}  n={len(g):>7,}  秩相关 {rc:.4f}  最大差 {np.abs(g.y_m - g.y_r).max():.2e}")
ex_t = ["09:45", "10:15", "10:45", "11:15", "13:15", "13:45", "14:15"]
pr = sec[["instrument"]].assign(day=sec["date"].dt.normalize()).drop_duplicates()
ex = pd.concat([pr.assign(date=pr["day"] + pd.Timedelta(f"{t}:00")) for t in ex_t], ignore_index=True)[["date", "instrument"]]
el = T._labels_from_bars(bars, ex)
el["hm"] = el["date"].dt.strftime("%H:%M")
print("中间时刻：覆盖率 / 标签标准差")
print(el.groupby("hm")["y"].agg(cover=lambda s: s.notna().mean(), std="std").round(5).to_string())
