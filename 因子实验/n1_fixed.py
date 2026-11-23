# -*- coding: utf-8 -*-
"""第十八轮 N1：把修复版近 5 日根（Y_sd5 / X_n5d）放回 1363，评估并交 values 三区间。

修复版根放在 因子实验/roots_src/r_{Y_sd5,X_n5d}_fixed.py：SQL 只做日内窗口（按股票×交易日聚合），
跨日滚动在 pandas 里做，平台不会踩到"跨日 1200 分钟窗口返回空"。
本脚本用 platform_sim 跑这两个根，映射到本地评测宇宙（engine.Core）的轴上，
再与平台实测的 1363 向量相加，得到"1363 + 修复版两根"的合成版。

用法：
  python 因子实验/n1_fixed.py --mode eval                 # 只本地评估五窗口
  python 因子实验/n1_fixed.py --mode submit --name v_1363FX # 评估 + 交 values 三区间
"""
import argparse
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/user/jyc_scorer/platform_sim")

import root_eval as R          # noqa: E402
import d2_optimize as O        # noqa: E402
from g_go import GO, inst_codes  # noqa: E402

DS = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
ROOTS = {"Y": HERE / "roots_src" / "r_Y_sd5_fixed.py",
         "X": HERE / "roots_src" / "r_X_n5d_fixed.py"}
WIN = {"main": ("2023-01-01 00:00:00", "2024-12-31 23:59:59"),
       "o21": ("2021-01-01 00:00:00", "2022-12-31 23:59:59"),
       "o20": ("2020-01-01 00:00:00", "2020-12-31 23:59:59")}
IX = {"main": 0, "o21": 1, "o20": 2}

_MOD = {}


def mod(path):
    if path not in _MOD:
        spec = importlib.util.spec_from_file_location("rootmod_" + path.stem, path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        _MOD[path] = m
    return _MOD[path]


_NEED = {}


def need_frame(c):
    if "f" not in _NEED:
        _NEED["f"] = pd.DataFrame({"row": np.arange(c.n_rows, dtype=np.int64),
                                   "date": pd.to_datetime(c.key_date),
                                   "instrument": inst_codes(c)})
    return _NEED["f"]


CACHE = pathlib.Path("/tmp/n1_cache")
CACHE.mkdir(exist_ok=True)


def root_axis(c, key, path, start, end):
    """把修复版根在 [start,end] 的输出映射成评测宇宙上的截面中心分位秩轴（带缓存）。"""
    cf = CACHE / ("%s_%s.npy" % (key, start[:7]))
    if cf.exists():
        return np.load(cf)
    df = mod(path).main(DS, start, end)
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    m = need_frame(c).merge(df, on=["date", "instrument"], how="left", sort=False)
    v = np.full(c.n_rows, np.nan)
    v[m["row"].to_numpy()] = m["factor"].to_numpy(float)
    ax = O.rc_of(v, c.date_id, c.n_dates)
    np.save(cf, ax)
    return ax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval")
    ap.add_argument("--name", default="v_1363FX")
    ap.add_argument("--wY", type=float, default=0.10)
    ap.add_argument("--wX", type=float, default=-0.10)
    ap.add_argument("--wins", default="main,o21,o20")
    a = ap.parse_args()

    G = GO()
    c = G.c
    print("评测宇宙：%d 行 / %d 个日期 / %d 只股票" % (c.n_rows, c.n_dates, len(np.unique(c.inst))))

    rows = []
    Bs = {}
    for win in [w.strip() for w in a.wins.split(",") if w.strip()]:
        st, en = WIN[win]
        ix = IX[win]
        axY = root_axis(c, "Y", ROOTS["Y"], st, en)
        axX = root_axis(c, "X", ROOTS["X"], st, en)
        base = np.asarray(G.B[("1363", ix)], float)
        # 单独版
        rows.append(("Y_sd5_fixed", np.where(np.isfinite(base), np.nan_to_num(axY, nan=0.0), np.nan), win, st, en))
        rows.append(("X_n5d_fixed", np.where(np.isfinite(base), np.nan_to_num(axX, nan=0.0), np.nan), win, st, en))
        # 放回 1363
        B = base + a.wY * np.nan_to_num(axY, nan=0.0) + a.wX * np.nan_to_num(axX, nan=0.0)
        Bs[win] = np.where(np.isfinite(base), B, np.nan)
        rows.append(("1363_fix", Bs[win], win, st, en))
        # 不做修复的对照：1363 原样
        rows.append(("1363_base", base, win, st, en))

    print("\n%-14s %-5s %9s %9s %9s %9s %9s | %s" % ("cand", "win", "IC", "IC_IR", "多空", "压力", "换手", "Δ vs 1363"))
    for nm, B, win, st, en in rows:
        m = G.evspan(B, st, en)
        r = G.evspan(np.asarray(G.B[("1363", IX[win])], float), st, en)
        d = "IC %+.5f IR %+.4f LS %+.2f 压力 %+.4f 换手 %+.4f" % (
            m["ic_mean"] - r["ic_mean"], m["ic_ir"] - r["ic_ir"], m["ls_sharpe"] - r["ls_sharpe"],
            m["stress"] - r["stress"], m["turnover"] - r["turnover"])
        print("%-14s %-5s %9.5f %9.4f %9.2f %9.4f %9.4f | %s" % (
            nm, win, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], d))

    if a.mode == "submit":
        G.write_submit(a.name, Bs, True)


if __name__ == "__main__":
    main()
