# -*- coding: utf-8 -*-
"""T2 的 5 分钟 patch 检查：数值分布 + 截断检测（只给截面时刻及之前的 1 分钟 bar，patch 必须逐位相同）。
用法：bash e2e/scripts/remote.sh e2e/scripts/s11_patch_check.py"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, set_read_cols  # noqa: E402

set_read_cols(T.FEATURE_COLS)
be = FeatherBackend()
lo, hi = "2023-03-06", "2023-03-10 23:59:59"
insts = be.pool(lo, hi)
sec = be.sections(lo, hi, insts).sort_values(["date", "instrument"]).reset_index(drop=True)
bars = be.bars("2023-02-27", hi, insts, T.FEATURE_COLS + ["adjust_factor"])
pc = T._patch_cube(bars, sec, 42)
hm = sec["date"].dt.strftime("%H%M").values
print("样本", len(sec), "patch", pc.shape)
for t in ["0930", "1000", "1130", "1330", "1430"]:
    m = hm == t
    v = np.isfinite(pc[m, :, 0]).sum(1)
    print(t, "有效 patch 数 中位", int(np.median(v)), "最大", int(v.max()))
names = ["L", "Lv", "amt", "r", "us", "bd"]
for j, n in enumerate(names):
    x = pc[..., j]
    print(f"{n:4s} nan={np.isnan(x).mean():.3f} mean={np.nanmean(x):+.4f} std={np.nanstd(x):.4f} p1/p99={np.nanpercentile(x, 1):+.4f}/{np.nanpercentile(x, 99):+.4f}")
bad = 0
for ts in ["2023-03-08 10:00:00", "2023-03-08 11:30:00", "2023-03-08 13:30:00", "2023-03-09 14:30:00"]:
    k = sec[sec["date"] == pd.Timestamp(ts)].reset_index(drop=True)
    a = T._patch_cube(bars, k, 42)
    b = T._patch_cube(bars[bars["date"] <= pd.Timestamp(ts)], k, 42)
    same = np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b))) == 0
    bad += not same
    print("截断检测", ts, "CLEAN" if same else "LEAK?", len(k))
print("T2 patch 检查", "通过" if bad == 0 else "失败")
