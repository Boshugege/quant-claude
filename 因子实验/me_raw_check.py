# -*- coding: utf-8 -*-
"""M2 代码版定权：底座用 1397（D2OPT3）的原始值（不是排名），看 原始值 + a·秩(收盘段季节性) + b·秩·[14:30] 的五项，
找出与研究结论（秩底座上 a=0.05、b=0.10）对应的原始值权重。年份 2023、2024、2021、2022。
运行：bash 工具/本地评分/remote.sh 因子实验/me_raw_check.py
"""
import multiprocessing as mp
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "工具/本地评分")
sys.path.insert(0, "jobs")
import localeval as le  # noqa: E402
import me_auction_study as A  # noqa: E402

YEARS = [2023, 2024, 2021, 2022]
A.YEARS = YEARS
WIN = {2023: "main", 2024: "main", 2021: "o21", 2022: "o21"}
P = None
GRID = [(0.0, 0.0), (0.05, 0.10), (0.075, 0.15), (0.094, 0.188), (0.12, 0.24)]


def build(year, F):
    p = le.build_panel(le.load_bars(f"{year}-01", f"{year}-12"))
    p = p.merge(F[["instrument_id", "day", "LAB_20"]], on=["instrument_id", "day"], how="left")
    kd = p["key_date"].to_numpy()
    d = pd.read_parquet(f"{A.BASE}/D2_{WIN[year]}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d["instrument"] = d["instrument"].astype(str).map(A.INV)
    d = d.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    m = p[["key_date", "instrument"]].merge(d.rename(columns={"date": "key_date"}), on=["key_date", "instrument"], how="left")
    p["raw"] = np.nan_to_num(m["factor"].to_numpy(float), nan=0.0)
    rk = pd.Series(p["LAB_20"].to_numpy(float)).groupby(kd).rank(pct=True).to_numpy() - 0.5
    p["x"] = np.where((p["slot"] != 930).to_numpy() & np.isfinite(rk), rk, 0.0)
    return p


def _ev(ab):
    a, b = ab
    s = P["slot"].to_numpy()
    v = P["raw"].to_numpy() + a * P["x"].to_numpy() + b * P["x"].to_numpy() * (s == 1430)
    m = le.evaluate(v, P)
    return (a, b, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"])


if __name__ == "__main__":
    F = A.day_table()
    for y in YEARS:
        P = build(y, F)
        with mp.get_context("fork").Pool(len(GRID)) as pool:
            res = pool.map(_ev, GRID)
        t = pd.DataFrame(res, columns=["a", "b", "IC", "IC_IR", "多空", "压力", "换手"])
        base = t.iloc[0]
        d = t.iloc[1:].copy()
        for k in ("IC", "IC_IR", "多空", "压力", "换手"):
            d[k] = d[k] - base[k]
        print(f"\n== {y} 底座 1397 原始值：IC {base['IC']:.5f} IR {base['IC_IR']:.4f} 多空 {base['多空']:.2f} 压力 {base['压力']:.4f} 换手 {base['换手']:.4f}；下表为差", flush=True)
        print(d.round(5).to_string(index=False), flush=True)
