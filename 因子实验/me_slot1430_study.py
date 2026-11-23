# -*- coding: utf-8 -*-
"""Claude 自主挖掘 M2b：14:30 截面加"收盘段季节性"，并试两种压换手的做法。
M2 发现：最后半小时 收盘 ÷ VWAP − 1 的过去 10 日均值（LAB_10，收盘集合竞价主导）在 14:30 截面单独 IC 0.095／0.076，
叠到 1398 的 14:30 截面上 IC／IC_IR／多空／压力两年都涨，但换手升（只改一个截面 → 14:00→14:30 那次换手变大）。
本脚本比较（特征 = 收盘段 LAB_N 的截面秩，同一交易日内取值不变）：
  只14:30 w        ：只在 14:30 截面加 w
  14:00半+14:30 w  ：14:00 加 w/2、14:30 加 w（把排名变化分摊到两次换手）
  全天0.05+14:30 w ：所有截面加 0.05、14:30 再加 w
底座：1398（G1B_B1）、1399（G4LOW）、1397（D2OPT3，只有主区间与 2021–22）。年份：2023、2024（样本外）与 2021、2022。
运行：bash 工具/本地评分/remote.sh 因子实验/me_slot1430_study.py
"""
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "工具/本地评分")
sys.path.insert(0, "jobs")
import localeval as le  # noqa: E402
import me_auction_study as A  # noqa: E402

YEARS = [int(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2023", "2024", "2021", "2022"])]
A.YEARS = YEARS
FILES = {"1398": "G1B", "1399": "G4L", "1397": "D2"}
WIN = {2023: "main", 2024: "main", 2021: "o21", 2022: "o21"}
P = None
FEAT = ["LAB_10", "LAB_20"]


def build(year, F):
    t0 = time.time()
    p = le.build_panel(le.load_bars(f"{year}-01", f"{year}-12"))
    p = p.merge(F[["instrument_id", "day"] + FEAT], on=["instrument_id", "day"], how="left")
    kd = p["key_date"].to_numpy()
    for nm, fl in FILES.items():
        d = pd.read_parquet(f"{A.BASE}/{fl}_{WIN[year]}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        d["instrument"] = d["instrument"].astype(str).map(A.INV)
        d = d.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
        m = p[["key_date", "instrument"]].merge(d.rename(columns={"date": "key_date"}), on=["key_date", "instrument"], how="left")
        rk = pd.Series(m["factor"].to_numpy(float)).groupby(kd).rank(pct=True).to_numpy() - 0.5
        p["b_" + nm] = np.where(np.isfinite(rk), rk, 0.0)
    ok = (p["slot"] != 930).to_numpy()
    for c in FEAT:
        rk = pd.Series(p[c].to_numpy(float)).groupby(kd).rank(pct=True).to_numpy() - 0.5
        p["r_" + c] = np.where(ok & np.isfinite(rk), rk, 0.0)
    print(f"[{year}] 面板 {len(p):,} 行，{time.time() - t0:.0f}s", flush=True)
    return p


def cands():
    out = []
    for b in FILES:
        out.append((b, "base", "", 0.0))
        for c in FEAT:
            for w in (0.1, 0.2, 0.35):
                for mode in ("只14:30", "14:00半+14:30", "全天0.05+14:30"):
                    out.append((b, mode, c, w))
    return out


def _ev(args):
    b, mode, c, w = args
    v = P["b_" + b].to_numpy().copy()
    if mode != "base":
        x = P["r_" + c].to_numpy()
        s = P["slot"].to_numpy()
        if mode == "只14:30":
            v = v + w * x * (s == 1430)
        elif mode == "14:00半+14:30":
            v = v + w * x * (s == 1430) + 0.5 * w * x * (s == 1400)
        else:
            v = v + 0.05 * x + w * x * (s == 1430)
    m = le.evaluate(v, P)
    return (b, mode, c, w, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], m["slots"].get(1430, np.nan))


if __name__ == "__main__":
    F = A.day_table()
    allr = []
    for y in YEARS:
        P = build(y, F)
        with mp.get_context("fork").Pool(12) as pool:
            res = pool.map(_ev, cands(), chunksize=1)
        t = pd.DataFrame(res, columns=["底座", "方式", "特征", "w", "IC", "IC_IR", "多空", "压力", "换手", "IC1430"])
        t["年"] = y
        allr.append(t)
        for b in FILES:
            s = t[t["底座"] == b].copy()
            base = s[s["方式"] == "base"].iloc[0]
            s = s[s["方式"] != "base"]
            for k in ("IC", "IC_IR", "多空", "压力", "换手", "IC1430"):
                s[k] = s[k] - base[k]
            print(f"\n== {y} 底座 {b}：base IC {base['IC']:.5f} IR {base['IC_IR']:.4f} 多空 {base['多空']:.2f} 压力 {base['压力']:.4f} 换手 {base['换手']:.4f} 14:30 IC {base['IC1430']:.4f}；下表为差", flush=True)
            print(s.drop(columns=["底座", "年"]).round(5).to_string(index=False), flush=True)
    pd.concat(allr).to_csv("工具/本地评分/结果/me_slot1430_study.csv", index=False)
