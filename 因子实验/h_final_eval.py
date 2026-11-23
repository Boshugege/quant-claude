# -*- coding: utf-8 -*-
"""第十四轮总结表：各候选在 main(2023-24)／o21(2021-22)／o20(2020) 三个区间的五项。"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402
from g_go import GO, _en  # noqa: E402
from h1_stack import build, g4low_delta  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
SPAN = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
        "o20": ("2020-01-01", "2020-12-31")}
YRS = ("2020", "2021", "2022", "2023", "2024")


def stitch(tag):
    c = R.core()
    yr = pd.to_datetime(c.key_date).year
    B = np.zeros(c.n_rows)
    for nm in YRS:
        v = np.load(OUT / ("h2_%s_%s.npy" % (tag, nm)))
        m = yr == int(nm)
        B[m] = v[m]
    return B


def main():
    G = GO()
    c = G.c
    d = g4low_delta()
    from h1_stack import D1B, BASES
    B = {
        "1363": {ix: np.asarray(G.B[("1363", ix)], float) for ix in (0, 1, 2)},
        "1398 G1B_B1": build(G, 0, "s161", D1B),
        "1399 G4LOW": {0: build(G, 0, "1363", d), 1: build(G, 1, "1363", d), 2: build(G, 2, "1363", d)},
        "1399+0.2L5": None,
        "v_H2GBDT": None,
        "v_H2SEAS": None,
    }
    B["1398 G1B_B1"] = {0: build(G, 0, "s161", D1B), 1: build(G, 1, "s161", D1B), 2: build(G, 2, "s161", D1B)}
    sig = np.nan_to_num(O.rc_of(np.load(OUT / "h1_L_5.npy"), c.date_id, c.n_dates), nan=0.0)
    B["1399+0.2L5"] = {ix: B["1399 G4LOW"][ix] + 0.2 * sig for ix in (0, 1, 2)}
    B["v_H2GBDT"] = {ix: stitch("gbdt") for ix in (0, 1, 2)}
    B["v_H2SEAS"] = {ix: stitch("seas") for ix in (0, 1, 2)}
    B["1397 D2OPT3"] = None
    print("%-13s %-5s %8s %8s %8s %8s %8s" % ("cand", "win", "IC", "IC_IR", "多空", "压力", "换手"))
    for nm, v in B.items():
        if v is None:
            continue
        for win, ix in (("main", 0), ("o21", 1), ("o20", 2)):
            st, en = SPAN[win]
            m, _ = c.evaluate(np.where(G.z930, 0.0, np.asarray(v[ix], float)), start=st, end=_en(en))
            print("%-13s %-5s %8.5f %8.4f %8.2f %8.4f %8.4f" %
                  (nm, win, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"]))


if __name__ == "__main__":
    main()
