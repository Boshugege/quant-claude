# -*- coding: utf-8 -*-
"""第十六轮 K1 交付：把收盘段季节性锚 CLS20 加到指定底座，交 values 三区间。

用法：python 因子实验/k1_submit.py --base 1399 --name v_CLS1399 --submit
权重与 K1 一致：a = 0.05·σ_底座/0.289（σ 为该底座原始值的截面标准差中位数），14:30 截面再加一份 b=a（共 2a）。
--no-boost 时不加 14:30 那一份（1398 上更优的版本）。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import k1_anchor as K  # noqa: E402
from g_go import GO  # noqa: E402
from h1_stack import build as h1build, D1B, g4low_delta  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="1399")
    ap.add_argument("--name", default="")
    ap.add_argument("--no-boost", action="store_true")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    G = GO()
    c = G.c
    kd = pd.to_datetime(c.key_date)
    z930 = c.slot_of_date[c.date_id] == 930
    slot = c.slot_of_date[c.date_id]
    D = K.load_frames()
    AN = K.make_anchors(D)
    _, r = K.rank_of(c, AN["CLS20"], z930)
    yr = kd.year.to_numpy()
    if a.base == "1397":
        B = {ix: np.asarray(K.R.load(("v_D2OPT3_raw", "v_D2OPT3O21_raw", "v_D2OPT3O20_raw")[ix])[0], float)
             for ix in (0, 1, 2)}
    elif a.base == "1398":
        B = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    elif a.base == "1399":
        B = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}
    elif a.base == "H2SEAS":
        H2 = np.zeros(c.n_rows)
        for y in ("2020", "2021", "2022", "2023", "2024"):
            v = np.load(OUT / ("h2_seas_%s.npy" % y))
            H2[yr == int(y)] = v[yr == int(y)]
        B = {ix: H2 for ix in (0, 1, 2)}
    else:
        raise SystemExit("未知底座 %s" % a.base)
    SPAN = {0: ("2023-01-01", "2024-12-31"), 1: ("2021-01-01", "2022-12-31"), 2: ("2020-01-01", "2020-12-31")}
    Bs = {}
    for wn, ix in (("main", 0), ("o21", 1), ("o20", 2)):
        sd = K.xsec_sd(c, B[ix], *SPAN[ix])
        aa = K.A_RANK * sd / K.RANK_SD
        x = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan)
        comp = x + aa * r + (0.0 if a.no_boost else aa) * r * (slot == 1430)
        Bs[wn] = np.where(z930, 0.0, comp)
        print("%s 区间 %s 权重 a=%.5f%s" % (a.base, wn, aa, "（不加14:30）" if a.no_boost else ""), flush=True)
    G.write_submit(a.name or ("v_CLS" + a.base), Bs, a.submit)


if __name__ == "__main__":
    main()
