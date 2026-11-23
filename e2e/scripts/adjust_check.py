# -*- coding: utf-8 -*-
"""复权检查：找除权日的 09:30 截面，比较复权前后窗口内（跨日）价格跳变。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, read_range  # noqa: E402

raw = read_range("2023-05-15", "2023-06-30 23:59:59")[["date", "instrument", "adjust_factor"] + T.FEATURE_COLS]
d = raw.assign(day=raw["date"].dt.normalize()).groupby(["instrument", "day"])["adjust_factor"].first()
chg = d.groupby(level=0).pct_change().abs()
ev = chg[chg > 0.02].reset_index().query("day >= '2023-06-01'").head(200)
sec = pd.DataFrame({"date": ev["day"] + pd.Timedelta("09:30:00"), "instrument": ev["instrument"]})
ci = T.FEATURE_COLS.index("close")
for name, b in [("未复权", raw.drop(columns="adjust_factor")), ("复权", raw)]:
    Xf, _, _, k = T.build_windows(b, sec, coarse_len=0)
    c = Xf[..., ci]
    jump = np.nanmax(np.abs(np.diff(np.log(c), axis=1)), axis=1)
    print(f"{name}: 除权日 09:30 截面 {len(k)} 个，窗口内最大 1 分钟对数跳变 中位数={np.nanmedian(jump):.4f} "
          f"90分位={np.nanpercentile(jump, 90):.4f}")
