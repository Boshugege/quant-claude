# -*- coding: utf-8 -*-
"""风格特征核对：提交脚本的 c_size/c_ivol/c_illiq/c_lowp 与因子赛道 factor_lib（size_neg/ivol/amihud/-ln px）逐截面秩相关。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
sys.path.insert(0, "工具/本地评分")
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, set_read_cols  # noqa: E402
import localeval as le  # noqa: E402
from factor_lib import add_components, size_neg, ivol, amihud  # noqa: E402

T.OUT_NEUTRAL = 1.5
set_read_cols(T.FEATURE_COLS)
fb = FeatherBackend()
lo, hi = "2023-03-01 00:00:00", "2023-03-31 23:59:59"
inst = fb.pool(lo, hi)
sec = fb.sections(lo, hi, inst)
bars = fb.bars("2023-02-09 00:00:00", hi, inst, T.FEATURE_COLS + ["adjust_factor"])
_, _, _, keys = T.build_windows(bars, sec)
p = add_components(le.build_panel(le.load_bars("2023-02", "2023-03")))
hh = (p["slot"] // 100).astype(int).astype(str).str.zfill(2) + ":" + (p["slot"] % 100).astype(int).astype(str).str.zfill(2)
p["date2"] = p["day"] + pd.to_timedelta(hh + ":00")
p["instrument"] = "ID" + p["instrument_id"].astype(int).astype(str).str.zfill(4)
ref = pd.DataFrame({"date": p["date2"], "instrument": p["instrument"], "f_size": size_neg(p), "f_ivol": ivol(p),
                    "f_illiq": amihud(p), "f_lowp": -np.log(p["px"])})
m = keys.merge(ref, on=["date", "instrument"])
print("matched", len(m), "of", len(keys), "| NaN 比例:", m[T.CHAR_COLS].isna().mean().round(4).to_dict())
for a, b in zip(T.CHAR_COLS, ["f_size", "f_ivol", "f_illiq", "f_lowp"]):
    rc = m.groupby("date").apply(lambda g: g[a].rank().corr(g[b].rank()))
    print(f"{a} vs {b}: 逐截面秩相关 均值 {rc.mean():.4f} 最小 {rc.min():.4f}")
