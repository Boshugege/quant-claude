# -*- coding: utf-8 -*-
"""校验 numpy 版 30m 聚合与 pandas resample_30m 逐值一致，并比较耗时。"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import read_range  # noqa: E402

T.set_fields("full")
bars = read_range("2021-02-08", "2021-03-31 23:59:59")[["date", "instrument"] + T.FEATURE_COLS]
inst = np.sort(bars["instrument"].unique())
grid, cube = T._cube(bars, "date", inst, T.FEATURE_COLS)
t0 = time.time()
cg, cc = T._coarse_from_cube(grid, cube)
t1 = time.time()
rb = T.resample_30m(bars)
pg, pc = T._cube(rb, "bucket", inst, T.FEATURE_COLS)
t2 = time.time()
print(f"numpy {t1 - t0:.1f}s  pandas {t2 - t1:.1f}s  grids equal={np.array_equal(cg, pg)} shape={cc.shape}")
a, b = np.nan_to_num(cc, nan=-9), np.nan_to_num(pc, nan=-9)
bad = ~np.isclose(a, b, rtol=1e-6, atol=1e-3)
print("mismatch cells:", int(bad.sum()), "of", bad.size, "by field:",
      {c: int(bad[..., k].sum()) for k, c in enumerate(T.FEATURE_COLS) if bad[..., k].any()})
