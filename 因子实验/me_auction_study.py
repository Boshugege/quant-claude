# -*- coding: utf-8 -*-
"""Claude 自主挖掘 M2：收盘集合竞价专精（只作用在 14:30 截面）。
14:30 截面的标签 = 15:00 收盘价 ÷ (14:30,15:00] 窗口 VWAP − 1，而 15:00 收盘价就是收盘集合竞价的成交价。
所以"个股收盘竞价的价格冲击习惯"（收盘价 ÷ 14:57 连续竞价最后价 − 1）可能比整段同时段季节性（L_lab10）更干净。
特征（截面日之前的交易日，按同一股票滚动再错开 1 日）：
  AUC_N  收盘竞价价格冲击的过去 N 日均值（N = 5／10／20）
  AVS_N  收盘竞价成交量占最后半小时成交量的过去 N 日均值
  LAB_N  最后半小时 收盘 ÷ VWAP − 1 的过去 N 日均值（= 14:30 截面的 L_labN，作对照）
评估：① 14:30 截面单独 IC；② 叠到底座（1398／1600）上，只改 14:30 截面：b + w × 截面秩(特征)，看五项与 14:30 的 IC。
运行：bash 工具/本地评分/remote.sh 因子实验/me_auction_study.py
"""
import glob
import multiprocessing as mp
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.feather as pf

sys.path.insert(0, "工具/本地评分")
import localeval as le  # noqa: E402

BASE = "jobs/basev"
D1M = "数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m"
YEARS = [int(a) for a in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["2023", "2024", "2021", "2022"])]
FILES = {"1398": "G1B", "1600": "H2S"}
WIN = {2023: "main", 2024: "main", 2021: "o21", 2022: "o21", 2020: "o20"}
idm = pd.read_csv(f"{BASE}/id_map.csv")
idm = idm[idm["ok"]]
INV = dict(zip(idm["instrument"], idm["local_id"]))
P = None


def day_table():
    """逐（股票，交易日）的收盘竞价统计，覆盖 YEARS 最早一年之前一个月起。"""
    lo = f"{min(YEARS) - 1}11"
    rows = []
    for f in sorted(glob.glob(f"{D1M}/*.0.feather")):
        if f.split("/")[-1][:6] < lo:
            continue
        t = pf.read_table(f, columns=["date", "instrument_id", "close", "volume", "amount"]).to_pandas()
        hm = t["date"].dt.hour * 100 + t["date"].dt.minute
        t = t[(hm > 1430) & (hm <= 1500)].copy()
        t["hm"] = (t["date"].dt.hour * 100 + t["date"].dt.minute).astype(int)
        t["day"] = t["date"].dt.normalize()
        t["close"] = t["close"].astype(float).where(t["close"] > 0)
        g = t.groupby(["instrument_id", "day"], sort=False)
        cont = t[(t["hm"] <= 1457) & (t["volume"] > 0)].groupby(["instrument_id", "day"], sort=False)["close"].last()
        auc = t[t["hm"] == 1500].set_index(["instrument_id", "day"])
        vol = g["volume"].sum()
        amt = g["amount"].sum()
        d = pd.DataFrame({"c_cont": cont, "c_close": auc["close"].where(auc["volume"] > 0),
                          "v_auc": auc["volume"].astype(float), "v_win": vol.astype(float), "a_win": amt.astype(float)})
        rows.append(d)
    d = pd.concat(rows).sort_index()
    vwap = d["a_win"] / d["v_win"].where(d["v_win"] > 0)              # 成交额 ÷ 成交量，与 close 同为"分"
    d["auc"] = d["c_close"] / d["c_cont"] - 1
    d["avs"] = d["v_auc"] / d["v_win"].where(d["v_win"] > 0)
    d["lab"] = d["c_close"] / vwap - 1
    d = d.replace([np.inf, -np.inf], np.nan)
    feats = {}
    g = d.groupby(level=0, sort=False)
    for col, nm in (("auc", "AUC"), ("avs", "AVS"), ("lab", "LAB")):
        for N in (5, 10, 20):
            m = g[col].rolling(N, min_periods=max(N // 2, 3)).mean()
            m.index = m.index.droplevel(0)
            feats[f"{nm}_{N}"] = m.reindex(d.index).groupby(level=0, sort=False).shift(1)
    f = pd.DataFrame(feats)
    print("日表", len(d), "auc 中位 %.5f 均值 %.5f；lab 中位 %.5f" % (d["auc"].median(), d["auc"].mean(), d["lab"].median()), flush=True)
    return f.reset_index().rename(columns={"level_0": "instrument_id", "level_1": "day"})


def build(year, F):
    t0 = time.time()
    p = le.build_panel(le.load_bars(f"{year}-01", f"{year}-12"))
    p = p.merge(F, on=["instrument_id", "day"], how="left")
    for nm, fl in FILES.items():
        d = pd.read_parquet(f"{BASE}/{fl}_{WIN[year]}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        d["instrument"] = d["instrument"].astype(str).map(INV)
        d = d.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
        m = p[["key_date", "instrument"]].merge(d.rename(columns={"date": "key_date"}), on=["key_date", "instrument"], how="left")
        rk = pd.Series(m["factor"].to_numpy(float)).groupby(p["key_date"].to_numpy()).rank(pct=True).to_numpy() - 0.5
        p["b_" + nm] = np.where(np.isfinite(rk), rk, 0.0)
    s1430 = (p["slot"] == 1430).to_numpy()
    for c in F.columns:
        if c in ("instrument_id", "day"):
            continue
        rk = pd.Series(p[c].to_numpy(float)).groupby(p["key_date"].to_numpy()).rank(pct=True).to_numpy() - 0.5
        p["r_" + c] = np.where(s1430 & np.isfinite(rk), rk, 0.0)
    print(f"[{year}] 面板 {len(p):,} 行，{time.time() - t0:.0f}s", flush=True)
    return p


FEATS = [f"{a}_{n}" for a in ("AUC", "AVS", "LAB") for n in (5, 10, 20)]


def cands():
    out = [("单独", c, 0.0) for c in FEATS]
    for b in FILES:
        out.append((b, None, 0.0))
        for c in ("AUC_5", "AUC_10", "AUC_20", "AVS_10", "LAB_10"):
            for w in (0.1, 0.25, 0.5):
                out.append((b, c, w))
            out.append((b, c, -0.25))
    return out


def _ev(args):
    b, c, w = args
    if b == "单独":
        v = P["r_" + c].to_numpy()
    else:
        v = P["b_" + b].to_numpy() + (w * P["r_" + c].to_numpy() if c else 0.0)
    m = le.evaluate(v, P)
    return (b, c or "base", w, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], m["slots"].get(1430, np.nan))


if __name__ == "__main__":
    F = day_table()
    allr = []
    for y in YEARS:
        P = build(y, F)
        with mp.get_context("fork").Pool(12) as pool:
            res = pool.map(_ev, cands(), chunksize=1)
        t = pd.DataFrame(res, columns=["底座", "特征", "w", "IC", "IC_IR", "多空", "压力", "换手", "IC1430"])
        t["年"] = y
        allr.append(t)
        print(f"\n== {y} 单独（只在 14:30 截面有值）的 14:30 IC", flush=True)
        print(t[t["底座"] == "单独"].set_index("特征")["IC1430"].round(5).to_string(), flush=True)
        for b in FILES:
            s = t[t["底座"] == b].copy()
            base = s[s["特征"] == "base"].iloc[0]
            s = s[s["特征"] != "base"]
            for k in ("IC", "IC_IR", "多空", "压力", "换手", "IC1430"):
                s[k] = s[k] - base[k]
            print(f"\n== {y} 底座 {b}：base IC {base['IC']:.5f} IR {base['IC_IR']:.4f} 多空 {base['多空']:.2f} 压力 {base['压力']:.4f} 换手 {base['换手']:.4f} 14:30 IC {base['IC1430']:.4f}；下表为差", flush=True)
            print(s.drop(columns=["底座", "年"]).round(5).to_string(index=False), flush=True)
    pd.concat(allr).to_csv("工具/本地评分/结果/me_auction_study.csv", index=False)
