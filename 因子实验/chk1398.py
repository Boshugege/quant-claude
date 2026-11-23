# -*- coding: utf-8 -*-
"""核对：平台实测的 1398（G1B_B1, 1017-123058-0dc6）与本地重建（h1build s161+D1B）是不是同一个向量。

--drop5d：D1B 里去掉 Y_sd5／X_n5d 两项（平台 w5 分钟窗口为空 → 代码版里这两项≡0），
用来验证"1398 本地重建与代码版秩相关只有 0.929"是不是就由这两项造成。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
from g_go import GO, _en  # noqa: E402
from h1_stack import build as h1build, D1B  # noqa: E402

SPANS = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"), "o20": ("2020-01-01", "2020-12-31")}
JOB = {"main": "1017-123058-0dc6", "o21": "1017-123111-5574", "o20": "1017-123118-e120"}
IX = {"main": 0, "o21": 1, "o20": 2}
KEY = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop5d", action="store_true")
    a = ap.parse_args()
    delta = dict(D1B)
    if a.drop5d:
        for k in ("Y_sd5", "X_n5d"):
            delta.pop(k, None)
        print("D1B 去掉 Y_sd5／X_n5d，剩 %d 项：%s" % (len(delta), delta), flush=True)
    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    for wn, (st, en) in SPANS.items():
        V = np.asarray(R.load(JOB[wn])[0], float)
        V = np.where(z930, 0.0, np.nan_to_num(V, nan=0.0))
        B = np.where(z930, 0.0, np.nan_to_num(h1build(G, IX[wn], "s161", delta), nan=0.0))
        mv, _ = c.evaluate(V, start=st, end=_en(en))
        mb, _ = c.evaluate(B, start=st, end=_en(en))
        ok = np.isfinite(V) & np.isfinite(B) & ~z930
        dm = np.zeros(c.n_dates, bool)
        dm[np.unique(c.date_id[ok])] = True
        rho = c.per_date_rank_corr(V, B, dm)
        print("%-4s 平台任务 %s | 平台任务向量本地重算 %s | 本地重建 %s | 秩相关 %.4f" % (
            wn, JOB[wn], "%.5f/%.4f/%.2f/%.4f/%.4f" % tuple(mv[k] for k in KEY),
            "%.5f/%.4f/%.2f/%.4f/%.4f" % tuple(mb[k] for k in KEY), rho), flush=True)


if __name__ == "__main__":
    main()
