# -*- coding: utf-8 -*-
"""面板缓存与“在面板上算因子”的模式（因子脚本 factor(panel)、factor_lib 已登记编号）。

全量 5 年面板加上 factor_lib 的中间量约 10 GB，计算同伴类因子时峰值超过 30 GB，远端只有 42 GB 内存。
所以按年分块计算：每块读取 [年初 − WARMUP_MONTHS, 年末] 的面板（由全量面板切出，昨收、标签、暴露都与全量一致），
算完只保留本年的评估行。6 个月的预热覆盖 factor_lib 中最长的 60 个交易日回看。
与 localeval 直接对 [start, end] 建面板相比，区间开头的因子值有完整历史（更接近平台：平台会多取历史数据）。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import localeval as le  # noqa: E402


def configure_localeval():
    """把逐字复制的 localeval 指向远端数据目录（不修改其源码）。"""
    le.DATA_DIR = str(C.DATA / "bar30m")
    le.EXPO_DIR = str(C.DATA / "exposure")
    le.LABEL_CACHE = str(C.DATA / "labels_y_off.parquet")
    le.ROOT = str(C.ROOT)


def save_panel(p: pd.DataFrame):
    C.PANEL_DIR.mkdir(parents=True, exist_ok=True)
    ym = p["day"].dt.strftime("%Y%m")
    for m, idx in p.groupby(ym).groups.items():
        p.loc[idx].reset_index(drop=True).to_parquet(C.PANEL_DIR / f"{m}.parquet", index=False)


def load_panel(start: str, end: str) -> pd.DataFrame:
    """读取 [start, end] 月份（含）的面板，排序与 build_panel 相同（instrument_id, date）。"""
    ms = [p.strftime("%Y%m") for p in pd.period_range(start, end, freq="M")]
    parts = [pd.read_parquet(C.PANEL_DIR / f"{m}.parquet") for m in ms if (C.PANEL_DIR / f"{m}.parquet").exists()]
    if not parts:
        raise FileNotFoundError(f"面板缓存没有 {start}~{end}，请先运行 prepare_data.py core")
    p = pd.concat(parts, ignore_index=True)
    return p.sort_values(["instrument_id", "date"]).reset_index(drop=True)


def year_chunks(start: str, end: str):
    s, e = pd.Period(start, "M"), pd.Period(end, "M")
    out = []
    for y in range(s.year, e.year + 1):
        a = max(s, pd.Period(f"{y}-01", "M"))
        b = min(e, pd.Period(f"{y}-12", "M"))
        out.append((str(a), str(b)))
    return out


def factor_on_panel(fn, start: str, end: str, log=print, components=True):
    """按年分块在面板上计算因子，返回评估行的 DataFrame[key_date, instrument_id, factor]。"""
    from factor_lib import add_components
    full_start = pd.Period(C.FULL_START, "M")
    outs = []
    for a, b in year_chunks(start, end):
        t0 = time.time()
        w = max(full_start, pd.Period(a, "M") - C.WARMUP_MONTHS)
        p = load_panel(str(w), b)
        if components:
            p = add_components(p)
        f = pd.Series(np.asarray(fn(p), dtype=np.float64), index=p.index)
        keep = p["is_eval"].to_numpy() & (p["day"] >= pd.Period(a, "M").start_time).to_numpy() & \
            (p["day"] <= pd.Period(b, "M").end_time).to_numpy()
        outs.append(pd.DataFrame({"key_date": p.loc[keep, "key_date"].to_numpy(),
                                  "instrument_id": p.loc[keep, "instrument_id"].to_numpy().astype(np.int32),
                                  "factor": f[keep].to_numpy()}))
        log(f"因子计算 {a}~{b}（预热自 {w}）：{keep.sum():,} 行，{time.time() - t0:.0f}s")
        del p, f
    return pd.concat(outs, ignore_index=True)
