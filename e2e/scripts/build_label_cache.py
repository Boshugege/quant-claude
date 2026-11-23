# -*- coding: utf-8 -*-
"""一次性生成 2020–2024 全部截面的本地标签缓存（y = 官方 vwap_return 口径，见 reports/12），供 FeatherBackend.labels 直接读取。"""
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import feather_data as F  # noqa: E402

t0 = time.time()
F.set_read_cols(F.LABEL_COLS)
parts = []
for m in F.month_list("2020-01", "2024-12"):
    d = F.read_month(m)
    parts.append(F.make_labels(d))
    print(m, len(parts[-1]), f"{time.time() - t0:.0f}s", flush=True)
lab = pd.concat(parts, ignore_index=True).sort_values(["date", "instrument"]).reset_index(drop=True)
os.makedirs(os.path.dirname(F.LABEL_CACHE), exist_ok=True)
lab.to_parquet(F.LABEL_CACHE)
print(f"saved {F.LABEL_CACHE} rows={len(lab):,} y_nonnull={lab['y'].notna().sum():,} in {time.time() - t0:.0f}s")
