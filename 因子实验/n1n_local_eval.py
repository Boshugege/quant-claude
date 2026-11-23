# -*- coding: utf-8 -*-
"""第十八轮补：**本地口径**评估 因子代码/*.py 的三区间五项（与评测机 platform 口径同一 core）。

用途：评测机队列很长（2–3 并发、每个约 40 分钟），把最要紧的几个候选在本机并行算掉。
用法：python 因子实验/n1n_local_eval.py 1761_xxx.py [1762_yyy.py ...] [--wins main,o21,o20]
"""
import argparse
import pathlib
import sys
import types

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/user/jyc_scorer/platform_sim")

import d2_optimize as O          # noqa: E402
from g_go import GO, inst_codes  # noqa: E402

FD = pathlib.Path("/home/user/quant-claude/因子代码")
DS = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments",
      "exposure": "cpt_jyc_2026_exposure"}
WINS = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
        "o20": ("2020-01-01", "2020-12-31")}


def axis(c, df):
    f = pd.DataFrame({"row": np.arange(c.n_rows, dtype=np.int64), "date": pd.to_datetime(c.key_date),
                      "instrument": inst_codes(c)})
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["instrument"] = d["instrument"].astype(str)
    m = f.merge(d, on=["date", "instrument"], how="left", sort=False)
    v = np.full(c.n_rows, np.nan)
    v[m["row"].to_numpy()] = m["factor"].to_numpy(float)
    return v


def run_one(G, name, wins):
    path = FD / name
    mod = types.ModuleType("m_" + path.stem)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), mod.__dict__)
    for w in wins:
        st, en = WINS[w]
        df = mod.main(DS, st + " 00:00:00", en + " 23:59:59")
        v = axis(G.c, df)
        m = G.evspan(v, st, en)
        print("%-46s %-5s IC %+.5f IR %+.4f 多空 %+6.2f 压力 %+.4f 换手 %.4f"
              % (name[:46], w, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"]), flush=True)
        del df, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--wins", default="main,o21,o20")
    a = ap.parse_args()
    wins = [x for x in a.wins.split(",") if x]
    G = GO()
    for nm in a.files:
        run_one(G, nm, wins)


if __name__ == "__main__":
    main()
