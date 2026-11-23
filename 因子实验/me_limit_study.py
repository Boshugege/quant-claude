# -*- coding: utf-8 -*-
"""Claude 自主挖掘 M1：涨跌停微观结构（新信息源）。
思路：标签 = 下一个 30 分钟窗口的 收盘 ÷ VWAP − 1。
- 封板（涨停且卖一为空 / 跌停且买一为空）的股票，下一个窗口价格被钉在板上，标签≈0（开板则偏负／偏正）。
  反转类分量会给它们极端分，可能是噪声。
- 接近涨停但未封板时，A 股常见"磁吸"（价格被吸向涨停）——窗口内持续上行 → 标签偏正；反转分量给它极端负分，方向可能错。
本脚本：① 分组诊断（各组占比、标签截面秩均值、底座因子截面秩均值）；② 几种简单修正在 2021–2024 各年的五项（平台口径 localeval）。
底座：1398（G1B_B1 values）、1600（H2SEAS values）、1363。运行：bash 工具/本地评分/remote.sh 因子实验/me_limit_study.py
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
FILES = {"1398": "G1B", "1600": "H2S", "1363": "V1363"}
WIN = {2023: "main", 2024: "main", 2021: "o21", 2022: "o21", 2020: "o20"}
idm = pd.read_csv(f"{BASE}/id_map.csv")
idm = idm[idm["ok"]]
INV = dict(zip(idm["instrument"], idm["local_id"]))
P = None


def build(year):
    t0 = time.time()
    p = le.build_panel(le.load_bars(f"{year - 1}-12", f"{year}-12"))
    p = p[p["day"].dt.year == year].reset_index(drop=True)
    code = p["instrument"].map(dict(zip(idm["local_id"], idm["instrument"])))
    # 板块涨跌幅：科创板 20%、创业板 2020-08-24 起 20%、其余 10%（ST 的 5% 不单独识别；封板判定不依赖它）
    cs = code.fillna("").astype(str)
    L = np.full(len(p), 0.1)
    L[cs.str.startswith("688").to_numpy()] = 0.2
    cy = cs.str.startswith(("300", "301")).to_numpy() & (p["day"] >= pd.Timestamp("2020-08-24")).to_numpy()
    L[cy] = 0.2
    pc = p["pre_close"].to_numpy(float)
    up = np.round(pc * (1 + L) + 1e-9, 2)
    dn = np.round(pc * (1 - L) + 1e-9, 2)
    cl = p["close"].to_numpy(float)
    ask1, bid1 = p["ask_price1"].to_numpy(float), p["bid_price1"].to_numpy(float)
    av1, bv1 = p["ask_volume1"].to_numpy(float), p["bid_volume1"].to_numpy(float)
    r = cl / pc - 1
    no_ask = ~(ask1 > 0) | ~(av1 > 0)
    no_bid = ~(bid1 > 0) | ~(bv1 > 0)
    p["lk_up"] = (r > 0.04) & no_ask                       # 涨停封板（含 ST 5%）
    p["lk_dn"] = (r < -0.04) & no_bid                      # 跌停封板
    p["d_up"] = up / cl - 1                                # 距涨停
    p["d_dn"] = cl / dn - 1                                # 距跌停
    p["r_day"] = r
    for nm, f in FILES.items():
        d = pd.read_parquet(f"{BASE}/{f}_{WIN[year]}.parquet")
        d["date"] = pd.to_datetime(d["date"])
        d["instrument"] = d["instrument"].astype(str).map(INV)
        d = d.dropna(subset=["instrument"]).drop_duplicates(["date", "instrument"])
        m = p[["key_date", "instrument"]].merge(d.rename(columns={"date": "key_date"}), on=["key_date", "instrument"], how="left")
        v = m["factor"].to_numpy(float)
        rk = pd.Series(v).groupby(p["key_date"].to_numpy()).rank(pct=True).to_numpy() - 0.5
        p["b_" + nm] = np.where(np.isfinite(rk), rk, 0.0)
    print(f"[{year}] 面板 {len(p):,} 行，{time.time() - t0:.0f}s", flush=True)
    return p


def diag(p, year):
    m = p["is_eval"].to_numpy() & (p["slot"] != 930).to_numpy() & np.isfinite(p["y_off"].to_numpy())
    q = p.loc[m, ["key_date", "y_off", "lk_up", "lk_dn", "d_up", "d_dn", "r_day"] + ["b_" + k for k in FILES]].copy()
    q["ry"] = q.groupby("key_date")["y_off"].rank(pct=True) - 0.5
    grp = np.select([q["lk_up"], q["lk_dn"],
                     (q["d_up"] <= 0.005) & ~q["lk_up"], (q["d_up"] > 0.005) & (q["d_up"] <= 0.02),
                     (q["d_up"] > 0.02) & (q["d_up"] <= 0.04),
                     (q["d_dn"] <= 0.005) & ~q["lk_dn"], (q["d_dn"] > 0.005) & (q["d_dn"] <= 0.02)],
                    ["1封涨停", "2封跌停", "3贴涨停未封", "4距涨停0.5-2%", "5距涨停2-4%", "6贴跌停未封", "7距跌停0.5-2%"], "9其他")
    q["g"] = grp
    t = q.groupby("g").agg(占比=("ry", "size"), 标签均值=("y_off", "mean"), 标签秩=("ry", "mean"),
                           **{f"{k}秩": ("b_" + k, "mean") for k in FILES})
    t["占比"] = t["占比"] / len(q)
    print(f"\n== {year} 分组诊断（非 09:30 评估截面）", flush=True)
    print(t.round(5).to_string(), flush=True)


def variants(p, base):
    b = p["b_" + base].to_numpy(float).copy()
    lu, ld = p["lk_up"].to_numpy(), p["lk_dn"].to_numpy()
    du, dd = p["d_up"].to_numpy(float), p["d_dn"].to_numpy(float)
    nu = np.clip(1 - du / 0.02, 0, 1) * (~lu)              # 越接近涨停越大（2% 以内），封板除外
    nd = np.clip(1 - dd / 0.02, 0, 1) * (~ld)
    nu, nd = np.nan_to_num(nu), np.nan_to_num(nd)
    out = {"base": b,
           "封板置0": np.where(lu | ld, 0.0, b),
           "涨停封板置0": np.where(lu, 0.0, b),
           "跌停封板置0": np.where(ld, 0.0, b)}
    for k in (0.15, 0.3):
        out[f"磁吸±{k}"] = np.where(lu | ld, 0.0, b) + k * nu - k * nd
        out[f"磁吸+{k}仅涨"] = np.where(lu | ld, 0.0, b) + k * nu
    return out


def _ev(args):
    year, base, name = args
    v = variants(P, base)[name]
    m = le.evaluate(v, P)
    return (year, base, name, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"])


if __name__ == "__main__":
    rows = []
    for y in YEARS:
        P = build(y)
        diag(P, y)
        names = list(variants(P, "1398").keys())
        tasks = [(y, b, n) for b in FILES for n in names]
        with mp.get_context("fork").Pool(min(10, len(tasks))) as pool:
            res = pool.map(_ev, tasks, chunksize=1)
        rows += res
        t = pd.DataFrame(res, columns=["年", "底座", "方案", "IC", "IC_IR", "多空", "压力", "换手"])
        for b in FILES:
            s = t[t["底座"] == b].set_index("方案")
            d = s[["IC", "IC_IR", "多空", "压力", "换手"]] - s.loc["base", ["IC", "IC_IR", "多空", "压力", "换手"]]
            print(f"\n== {y} 底座 {b}（base 绝对值：IC {s.loc['base', 'IC']:.5f} IR {s.loc['base', 'IC_IR']:.4f} 多空 {s.loc['base', '多空']:.2f} 压力 {s.loc['base', '压力']:.4f} 换手 {s.loc['base', '换手']:.4f}）", flush=True)
            print(d.drop(index="base").round(5).to_string(), flush=True)
    pd.DataFrame(rows, columns=["年", "底座", "方案", "IC", "IC_IR", "多空", "压力", "换手"]).to_csv("工具/本地评分/结果/me_limit_study.csv", index=False)
