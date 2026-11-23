# -*- coding: utf-8 -*-
"""第十七轮 L2：把两个 H2 版本按评测机"三区间"口径对比（相对 v_H2SEAS）。

用法：python 因子实验/h2_cmp.py --a h2_seas --b h2_upg_anch,h2_upg_simple
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
WIN = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"), "o20": ("2020-01-01", "2020-12-31")}
KEY = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")


def stack(c, tag):
    kd = pd.to_datetime(c.key_date)
    yr = kd.year.to_numpy()
    v = np.zeros(c.n_rows)
    for y in (2020, 2021, 2022, 2023, 2024):
        x = np.load(OUT / ("h2_%s_%d.npy" % (tag, y)))
        v[yr == y] = x[yr == y]
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="h2_seas")
    ap.add_argument("--cands", default="h2_upg_anch,h2_upg_simple")
    a = ap.parse_args()
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    B = np.nan_to_num(stack(c, a.base), nan=0.0)
    B[z930] = 0.0
    bm = {}
    for wn, (st, en) in WIN.items():
        bm[wn] = c.evaluate(B, start=st, end=pd.Timestamp(en) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))[0]
        print("%-4s 底座 %s IC %.5f IR %.4f 多空 %.2f 压力 %.4f 换手 %.4f" % (
            wn, a.base, *[bm[wn][k] for k in KEY]), flush=True)
    for tag in a.cands.split(","):
        v = np.nan_to_num(stack(c, tag), nan=0.0)
        v[z930] = 0.0
        print("\n=== %s vs %s ===" % (tag, a.base), flush=True)
        okall = True
        for wn, (st, en) in WIN.items():
            m = c.evaluate(v, start=st, end=pd.Timestamp(en) + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1))[0]
            d = [m[k] - bm[wn][k] for k in KEY]
            ok = all(x >= -1e-9 for x in d[:4]) and d[4] <= 1e-9
            okall &= ok
            print("  %-4s %s | Δ %+.5f/%+.4f/%+.2f/%+.4f/%+.4f %s" % (
                wn, "%.5f/%.4f/%.2f/%.4f/%.4f" % tuple(m[k] for k in KEY), *d, "★" if ok else ""), flush=True)
        print("  -> %s" % ("★ 三区间五项全好" if okall else "×"), flush=True)


if __name__ == "__main__":
    main()
