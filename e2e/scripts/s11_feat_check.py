# -*- coding: utf-8 -*-
"""S11 新输入（T1 同时段 token、T3 日线 60 天、T5 截面排名）检查：数值分布 + 截断检测（只给截面时刻及之前的窗口，特征必须逐位相同）。

用法：bash e2e/scripts/remote.sh e2e/scripts/s10_feat_check.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend  # noqa: E402

T.WIN_GROUPS = "shape,book,xrank"
T.E04_CFG.update(daily_len=60, slot_len=20)
base = ["L", "Lv", "amt", "r", "us", "bd", "at"] + list(T.BOOK30_COLS)
names = base + ["xr_" + c for c in base]
assert len(names) == T.n_win_feat(), (len(names), T.n_win_feat())

be = FeatherBackend()
lo, hi = "2023-03-01", "2023-03-10 23:59:59"
insts = be.pool(lo, hi)
sec = be.sections(lo, hi, insts).sort_values(["date", "instrument"]).reset_index(drop=True)
win = be.windows30(T._fmt(pd.Timestamp(lo) - pd.Timedelta(days=T.WIN_LOOKBACK_DAYS)), hi, insts)
bk = be.book30(T._fmt(pd.Timestamp(lo) - pd.Timedelta(days=T.BOOK_LOOKBACK_DAYS)), hi, insts)
win = win.merge(bk, on=["date", "instrument"], how="left")

Xi, Xd, Xp, ch = T._e04_window_feats(win, sec)
print(f"样本 {len(sec):,}  Xi {Xi.shape}  Xd {Xd.shape}")
assert Xi.shape[1] == 18 + 40 and Xd.shape[1] == 60, (Xi.shape, Xd.shape)
for nm in ["L", "Lv", "amt", "r", "b_imb", "xr_L", "xr_r"]:
    j = names.index(nm)
    a, b, c = Xi[:, 17, j], Xi[:, 18:, j], Xd[:, 0, j]
    print(f"{nm:6s} 最后日内 token nan={np.isnan(a).mean():.3f} std={np.nanstd(a):.4f} | 同时段 token nan={np.isnan(b).mean():.3f} "
          f"std={np.nanstd(b):.4f} | 最早日线 token nan={np.isnan(c).mean():.3f}")
bad = 0
for ts in ["2023-03-08 09:30:00", "2023-03-08 10:30:00", "2023-03-08 13:30:00", "2023-03-09 14:30:00"]:
    k = sec[sec["date"] == pd.Timestamp(ts)].reset_index(drop=True)
    full = T._e04_window_feats(win, k)
    trunc = T._e04_window_feats(win[win["date"] <= pd.Timestamp(ts)], k)
    for a, b, nm in zip(full, trunc, ["Xi", "Xd", "Xp", "ch"]):
        same = np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(np.nan_to_num(a) - np.nan_to_num(b))) == 0
        if not same:
            bad += 1
            d = np.abs(np.nan_to_num(a) - np.nan_to_num(b))
            print(f"  截断不一致 {ts} {nm}: 最大差 {d.max():.3e}，通道 {np.unique(np.nonzero(d.reshape(-1, d.shape[-1]) > 0)[1])}")
    print(f"截断检测 {ts}: {'CLEAN' if not bad else 'LEAK?'} (n={len(k)})")
print("S11 特征检查", "通过" if bad == 0 else "失败")
