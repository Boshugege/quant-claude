# -*- coding: utf-8 -*-
"""摘要特征检查：(1) 用 pandas 独立重算 r30 / r_yclose / r_dayvwap / pos_day 逐值对照；(2) 各特征单月秩 IC。"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, set_read_cols, read_range  # noqa: E402

T.set_fields(sys.argv[1] if len(sys.argv) > 1 else "full")
T.SUMMARY, T.COARSE_LEN = True, 0
set_read_cols(T.FEATURE_COLS)
fb = FeatherBackend()
lo, hi = "2023-03-01 00:00:00", "2023-03-31 23:59:59"
inst = fb.pool(lo, hi)
sec = fb.sections(lo, hi, inst)
raw = fb.bars("2023-02-09 00:00:00", hi, inst, T.FEATURE_COLS + ["adjust_factor"])
Xf, Xc, Xs, keys = T.build_windows(raw, sec)
bars = raw.copy()
for c in T.FEATURE_COLS:
    if T._is_price(c):
        bars[c] = bars[c] * bars["adjust_factor"]
    elif c == "volume" or "_volume" in c:
        bars[c] = bars[c] / bars["adjust_factor"]
names = T.summary_names()
print("Xs shape", Xs.shape, "nan frac per feature:", dict(zip(names, np.round(np.isnan(Xs).mean(0), 3))))

# (1) 独立重算
grid = np.sort(bars["date"].unique())
by = {k: g.set_index("date").sort_index().reindex(grid) for k, g in bars.groupby("instrument")}
rng = np.random.RandomState(1)
errs = {k: [] for k in ("r30", "r_yclose", "r_dayvwap", "pos_day")}
for i in rng.choice(len(keys), 300, replace=False):
    t, ins = keys.loc[i, "date"], keys.loc[i, "instrument"]
    g = by[ins]
    g = g[g.index <= t]
    mid = np.where((g.bid_price1 > 0) & (g.ask_price1 > g.bid_price1), (g.bid_price1 + g.ask_price1) / 2, np.nan)
    px = pd.Series(np.where(np.isfinite(mid), mid, g.close.where(g.close > 0)), index=g.index).ffill()
    cf = g.close.where(g.close > 0).ffill()
    p0 = px.iloc[-1]
    today = g.index.normalize() == pd.Timestamp(t).normalize()
    gt = g[today]
    exp = {"r30": np.log(p0 / px.iloc[-31]),
           "r_yclose": np.log(p0 / cf[~today].iloc[-1]),
           "r_dayvwap": np.log(p0 / (gt.amount.fillna(0).sum() / gt.volume.fillna(0).sum())),
           "pos_day": (cf.iloc[-1] - gt.low.min()) / (gt.high.max() - gt.low.min()) if gt.high.max() > gt.low.min() else 0.5}
    for k, v in exp.items():
        got = Xs[i, names.index(k)]
        if np.isfinite(v) or np.isfinite(got):
            errs[k].append(abs(float(v) - float(got)) if np.isfinite(v) and np.isfinite(got) else np.inf)
print("max abs diff vs independent pandas:", {k: (max(v) if v else None) for k, v in errs.items()})

# (2) 单月秩 IC（官方标签口径）
lab = fb.labels(lo, hi, inst)
d = keys.merge(lab[["date", "instrument", "y"]], on=["date", "instrument"], how="left")
Z = T.cs_zscore(Xs, keys["date"].values)
res = {}
for j, k in enumerate(names):
    df = pd.DataFrame({"d": d["date"], "f": Z[:, j], "y": d["y"]}).dropna()
    res[k] = df.groupby("d").apply(lambda g: g["f"].rank().corr(g["y"].rank())).mean()
print("rank IC 2023-03:", {k: round(v, 4) for k, v in res.items()})
