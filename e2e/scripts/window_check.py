# -*- coding: utf-8 -*-
"""切窗抽查：1m 窗口 = 全局网格上截面时刻（含）之前最近 64 根；30m 窗口 = 独立重算的 30 分钟 bar 中结束时刻 <= 截面的最近 COARSE_LEN 根。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, read_range  # noqa: E402

T.set_fields(sys.argv[1] if len(sys.argv) > 1 else "full")
L, LC = T.SEQ_LEN, T.COARSE_LEN
raw = read_range("2020-02-01", "2020-03-31 23:59:59")[["date", "instrument", "adjust_factor"] + T.FEATURE_COLS]
bars = raw.copy()                                   # 独立实现的复权：价格 × 因子，股数 ÷ 因子
for c in T.FEATURE_COLS:
    if T._is_price(c):
        bars[c] = bars[c] * bars["adjust_factor"]
    elif c == "volume" or "_volume" in c:
        bars[c] = bars[c] / bars["adjust_factor"]
fb = FeatherBackend()
sec = fb.sections("2020-03-01", "2020-03-31 23:59:59", fb.pool("2020-03-01", "2020-03-31"))
Xf, Xc, _, keys = T.build_windows(raw, sec)
grid = np.sort(bars["date"].unique())
by = {k: g.set_index("date").sort_index() for k, g in bars.groupby("instrument")}   # 文件内行不按时间排序
ci = {c: T.FEATURE_COLS.index(c) for c in ("open", "high", "low", "close", "volume", "bid_price1")}


def manual_30m(g):
    """独立实现：按 30 分钟窗口（集合竞价归 09:30）聚合。"""
    t = g.index
    hm = t.strftime("%H:%M")
    edges = ["09:30", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]
    lab = [next(e for e in edges if h <= e) for h in hm]
    end = pd.to_datetime(t.strftime("%Y-%m-%d") + " " + pd.Index(lab) + ":00")
    d = g.assign(end=end)
    return d.groupby("end").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                                close=("close", "last"), volume=("volume", "sum"), bid_price1=("bid_price1", "last"))


rng = np.random.RandomState(0)
bad_f = bad_c = 0
cache = {}
for i in rng.choice(len(keys), 200, replace=False):
    t, ins = keys.loc[i, "date"], keys.loc[i, "instrument"]
    e = np.searchsorted(grid, np.datetime64(t), side="right")
    ref = by[ins].reindex(grid[e - L:e])[T.FEATURE_COLS].to_numpy(np.float32)
    bad_f += not np.allclose(np.nan_to_num(ref, nan=-9), np.nan_to_num(Xf[i], nan=-9), rtol=1e-6)
    if ins not in cache:
        cache[ins] = manual_30m(by[ins])
    m = cache[ins][cache[ins].index <= t].iloc[-LC:]
    got = Xc[i][-len(m):]
    exp = m[list(ci)].to_numpy(np.float32)
    ok = np.allclose(np.nan_to_num(exp, nan=-9), np.nan_to_num(got[:, list(ci.values())], nan=-9), rtol=1e-5)
    bad_c += not ok
print(f"fields={T.FIELD_SET} checked 200 sections: 1m mismatches={bad_f}, 30m mismatches={bad_c}")
ex = keys.index[(keys["date"].dt.strftime("%H:%M") == "09:30")][0]
m = cache.get(keys.loc[ex, "instrument"]) if keys.loc[ex, "instrument"] in cache else manual_30m(by[keys.loc[ex, "instrument"]])
m = m[m.index <= keys.loc[ex, "date"]].iloc[-LC:]
print("09:30 example", keys.loc[ex].tolist(), "30m window:", m.index[0], "->", m.index[-1], "n =", len(m))
