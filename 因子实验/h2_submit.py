# -*- coding: utf-8 -*-
"""把 H2 的 GBDT 叠合（2020-22 拟合、逐年预测）按三区间提交成 values 任务。"""
import argparse
import pathlib
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g_go import GO  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
YRS = ("2020", "2021", "2022", "2023", "2024")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="gbdt")
    ap.add_argument("--name", default="v_H2GBDT")
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    G = GO()
    c = G.c
    yr = pd.to_datetime(c.key_date).year
    B = np.zeros(c.n_rows)
    for nm in YRS:
        p = OUT / ("h2_%s_%s.npy" % (a.tag, nm))
        v = np.load(p)
        m = yr == int(nm)
        B[m] = v[m]
        print("%s 行 %d 非零 %d" % (nm, m.sum(), (v[m] != 0).sum()))
    Bs = {"main": B, "o21": B, "o20": B}
    G.write_submit(a.name, Bs, a.submit)


if __name__ == "__main__":
    main()
