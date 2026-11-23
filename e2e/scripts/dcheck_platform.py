# -*- coding: utf-8 -*-
"""本地 1 分钟数据（训练用，feather）与平台 1 分钟数据（评测机 store，平台下载）逐项对比，并对比 5 分钟 patch。
用法：bash e2e/scripts/remote.sh e2e/scripts/dcheck_platform.py（先把平台月文件放到 jobs/dcheck/plat_YYYYMM.parquet）"""
import os, sys
import numpy as np
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, set_read_cols  # noqa: E402

set_read_cols(T.FEATURE_COLS)
cols = list(dict.fromkeys(T.FEATURE_COLS + ["deal_number"]))
plat = pd.read_parquet("jobs/dcheck/plat_202403.parquet", columns=["date", "instrument_id"] + [c for c in cols if c != "deal_number"])
plat["instrument"] = "ID" + plat["instrument_id"].astype(int).astype(str).str.zfill(4)
plat["date"] = pd.to_datetime(plat["date"])
be = FeatherBackend()
insts = sorted(set(be.pool("2024-03-01", "2024-03-31")) & set(plat["instrument"]))
loc = be.bars("2024-03-01", "2024-03-31 23:59:59", insts, T.FEATURE_COLS)
plat = plat[plat["instrument"].isin(insts)]
for name, df in (("平台", plat), ("本地", loc)):
    hm = df["date"].dt.strftime("%H%M")
    per = df.groupby([df["instrument"], df["date"].dt.normalize()]).size()
    print(f"{name}: 行 {len(df):,}  不同分钟 {hm.nunique()}  最早/最晚 {hm.min()}/{hm.max()}  每股每日 bar 数 中位 {per.median():.0f} 范围 {per.min()}–{per.max()}")
    print(f"   {name} 0925–0931 的分钟：{sorted(hm[(hm >= '0925') & (hm <= '0931')].unique())}  1129–1302：{sorted(hm[(hm >= '1129') & (hm <= '1302')].unique())}  1456–1500：{sorted(hm[hm >= '1456'].unique())}")
m = loc.merge(plat, on=["date", "instrument"], suffixes=("_l", "_p"))
print(f"按 (分钟, 股票) 对齐：{len(m):,} 行（本地 {len(loc):,}／平台 {len(plat):,}）")
for c in ("open", "close", "high", "volume", "amount", "bid_price1", "bid_volume1"):
    a, b = m[c + "_l"].astype(float), m[c + "_p"].astype(float)
    ok = (b != 0) & np.isfinite(a) & np.isfinite(b)
    r = a[ok] / b[ok]
    print(f"  {c:12s} 本地/平台 中位 {r.median():.4g}  1%/99% {r.quantile(0.01):.4g}/{r.quantile(0.99):.4g}  两边恰好相等比例 {(a[ok] == b[ok]).mean():.3f}")
sec = be.sections("2024-03-11", "2024-03-15 23:59:59", insts).sort_values(["date", "instrument"]).reset_index(drop=True)
lo, hi = pd.Timestamp("2024-03-04"), pd.Timestamp("2024-03-15 23:59:59")
pl = T._patch_cube(plat[(plat["date"] >= lo) & (plat["date"] <= hi)], sec, 42)
pc = T._patch_cube(loc[(loc["date"] >= lo) & (loc["date"] <= hi)], sec, 42)
names = ["L", "Lv", "amt", "r", "us", "bd"]
for j, n in enumerate(names):
    a, b = pc[..., j], pl[..., j]
    both = np.isfinite(a) & np.isfinite(b)
    print(f"patch {n:4s} 本地有效 {np.isfinite(a).mean():.3f} 平台有效 {np.isfinite(b).mean():.3f} 都有效时 最大差 {np.nanmax(np.abs(a[both] - b[both])):.3e} 相关 {np.corrcoef(a[both], b[both])[0, 1]:.4f}")
