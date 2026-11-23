# -*- coding: utf-8 -*-
"""Claude 自主挖掘 M2c：14:30 截面的日内动量（资料库待做⑤；Gao, Han, Li & Zhou 2018；A 股 Chu, Gu & Zhou 2019）。
假设：当日首半小时收益（或当日至今收益）正向预测最后半小时（14:30 截面的标签窗口，含收盘集合竞价）。
特征（当日、截面之前即可得到）：
  R_FH   首半小时收益 = 10:00 收盘 ÷ 09:25 竞价价 − 1（不含隔夜）
  R_ON   隔夜收益 = 09:25 竞价价 ÷ 昨收 − 1
  R_DAY  当日至今收益 = 14:30 收盘 ÷ 昨收 − 1
  R_PM   午后收益 = 14:30 收盘 ÷ 11:30 收盘 − 1
  R_L30  最近半小时收益 = 14:30 收盘 ÷ 14:00 收盘 − 1
评估：14:30 截面单独 IC；叠到 1398 的 14:30 截面（只改 14:30），正负两个方向。年份 2023、2024、2021、2022。
运行：bash 工具/本地评分/remote.sh 因子实验/me_mom1430_study.py
"""
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "工具/本地评分")
import localeval as le  # noqa: E402

BASE = "jobs/basev"
YEARS = [int(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2023", "2024", "2021", "2022"])]
WIN = {2023: "main", 2024: "main", 2021: "o21", 2022: "o21"}
idm = pd.read_csv(f"{BASE}/id_map.csv")
idm = idm[idm["ok"]]
INV = dict(zip(idm["instrument"], idm["local_id"]))
FEATS = ["R_FH", "R_ON", "R_DAY", "R_PM", "R_L30"]
P = None


def build(year):
    t0 = time.time()
    p = le.build_panel(le.load_bars(f"{year - 1}-12", f"{year}-12"))
    p = p[p["day"].dt.year == year].reset_index(drop=True)
    w = p.pivot_table(index=["instrument_id", "day"], columns="slot", values="close", aggfunc="last")
    pc = p.groupby(["instrument_id", "day"])["pre_close"].first()
    f = pd.DataFrame(index=w.index)
    f["R_FH"] = w[1000] / w[930] - 1
    f["R_ON"] = w[930] / pc - 1
    f["R_DAY"] = w[1430] / pc - 1
    f["R_PM"] = w[1430] / w[1130] - 1
    f["R_L30"] = w[1430] / w[1400] - 1
    f = f.replace([np.inf, -np.inf], np.nan).reset_index()
    p = p.merge(f, on=["instrument_id", "day"], how="left")
    kd = p["key_date"].to_numpy()
    s1430 = (p["slot"] == 1430).to_numpy()
    for c in FEATS:
        rk = pd.Series(p[c].to_numpy(float)).groupby(kd).rank(pct=True).to_numpy() - 0.5
        p["r_" + c] = np.where(s1430 & np.isfinite(rk), rk, 0.0)
    d = pd.read_parquet(f"{BASE}/G1B_{WIN[year]}.parquet")
    d["date"] = pd.to_datetime(d["date"])
    d["instrument"] = d["instrument"].astype(str).map(INV)
    d = d.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
    m = p[["key_date", "instrument"]].merge(d.rename(columns={"date": "key_date"}), on=["key_date", "instrument"], how="left")
    rk = pd.Series(m["factor"].to_numpy(float)).groupby(kd).rank(pct=True).to_numpy() - 0.5
    p["b_1398"] = np.where(np.isfinite(rk), rk, 0.0)
    print(f"[{year}] 面板 {len(p):,} 行，{time.time() - t0:.0f}s", flush=True)
    return p


def cands():
    out = [("单独", c, 0.0) for c in FEATS] + [("1398", "base", 0.0)]
    for c in FEATS:
        for w in (0.1, 0.25, -0.1, -0.25):
            out.append(("1398", c, w))
    return out


def _ev(args):
    b, c, w = args
    if b == "单独":
        v = P["r_" + c].to_numpy()
    else:
        v = P["b_1398"].to_numpy() + (w * P["r_" + c].to_numpy() if c != "base" else 0.0)
    m = le.evaluate(v, P)
    return (b, c, w, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], m["slots"].get(1430, np.nan))


if __name__ == "__main__":
    for y in YEARS:
        P = build(y)
        with mp.get_context("fork").Pool(12) as pool:
            res = pool.map(_ev, cands(), chunksize=1)
        t = pd.DataFrame(res, columns=["底座", "特征", "w", "IC", "IC_IR", "多空", "压力", "换手", "IC1430"])
        print(f"\n== {y} 单独的 14:30 IC", flush=True)
        print(t[t["底座"] == "单独"].set_index("特征")["IC1430"].round(5).to_string(), flush=True)
        s = t[t["底座"] == "1398"].copy()
        base = s[s["特征"] == "base"].iloc[0]
        s = s[s["特征"] != "base"]
        for k in ("IC", "IC_IR", "多空", "压力", "换手", "IC1430"):
            s[k] = s[k] - base[k]
        print(f"== {y} 底座 1398：base IC {base['IC']:.5f} 14:30 IC {base['IC1430']:.4f} 换手 {base['换手']:.4f}；下表为差", flush=True)
        print(s.drop(columns=["底座"]).round(5).to_string(index=False), flush=True)
