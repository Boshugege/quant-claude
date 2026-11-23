# -*- coding: utf-8 -*-
"""第十六轮 K2 交付：1399 底座 + 收盘段锚 CLS20（14:30 再加一份）+ CLOSEPOS20 锚（全天小权重）。

用法：python 因子实验/k2_submit.py --name v_CLSCP --submit
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import k1_anchor as K  # noqa: E402
from g_go import GO  # noqa: E402
from h1_stack import build as h1build, g4low_delta  # noqa: E402

SPAN = {0: ("2023-01-01", "2024-12-31"), 1: ("2021-01-01", "2022-12-31"), 2: ("2020-01-01", "2020-12-31")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v_CLSCP")
    ap.add_argument("--close-a", type=float, default=0.05)
    ap.add_argument("--cp-a", type=float, default=0.05)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    slot = c.slot_of_date[c.date_id]
    D = K.load_frames()
    AN = K.make_anchors(D)
    _, rcls = K.rank_of(c, AN["CLS20"], z930)
    _, rcp = K.rank_of(c, AN["CLOSEPOS20"], z930)
    B = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}
    Bs = {}
    for wn, ix in (("main", 0), ("o21", 1), ("o20", 2)):
        sd = K.xsec_sd(c, B[ix], *SPAN[ix])
        ac = a.close_a * sd / K.RANK_SD
        ap_ = a.cp_a * sd / K.RANK_SD
        x = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan)
        Bs[wn] = np.where(z930, 0.0, x + ac * rcls + ac * rcls * (slot == 1430) + ap_ * rcp)
        print("%s 区间 %s CLS20 a=%.4f CLOSEPOS20 a=%.4f" % ("1399", wn, ac, ap_), flush=True)
    G.write_submit(a.name, Bs, a.submit)


if __name__ == "__main__":
    main()
