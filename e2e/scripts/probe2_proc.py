# -*- coding: utf-8 -*-
"""反推平台的因子处理：对原始因子尝试多种 去极值 → 标准化 → 风格剔除 组合，看哪种与平台 process_factor 逐值一致。
用法（本机）：python3 e2e/scripts/probe2_proc.py
"""
import numpy as np
import pandas as pd

D = "数据/平台探测/"
STYLE = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
ex = pd.read_parquet("数据/exposure/exposure_2024.parquet")
IND = [c for c in ex.columns if c not in STYLE + ["date", "instrument", "industry_level1_code", "float_market_cap", "weights", "ret"]]
ex["day"] = pd.to_datetime(ex["date"]).dt.normalize()


def clip(x, how):
    if how == "none":
        return x
    med = np.median(x); mad = np.median(np.abs(x - med))
    if how.startswith("madraw"):
        k = float(how[6:]); return np.clip(x, med - k * mad, med + k * mad)
    if how.startswith("mad"):
        k = float(how[3:]); return np.clip(x, med - k * 1.4826 * mad, med + k * 1.4826 * mad)
    if how.startswith("sd"):
        k = float(how[2:]); mu, s = x.mean(), x.std(); return np.clip(x, mu - k * s, mu + k * s)
    if how.startswith("q"):
        p = float(how[1:]); lo, hi = np.percentile(x, [p, 100 - p]); return np.clip(x, lo, hi)


def proc(g, how, ind=False, std_after=False):
    x = g["factor"].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
    x = clip(x, how)
    if x.std() == 0:
        return np.full(len(x), np.nan)
    z = (x - x.mean()) / x.std()
    cols = STYLE + (IND if ind else [])
    B = g[cols].to_numpy(float)
    X = np.column_stack([np.ones(len(z)), B]) if not ind else B if False else np.column_stack([np.ones(len(z)), B])
    c, *_ = np.linalg.lstsq(X, z, rcond=None)
    r = z - X @ c
    if std_after:
        r = (r - r.mean()) / r.std()
    return r


for fid in ["90", "104"]:
    raw = pd.read_parquet(D + f"probe2_raw_factor_{fid}.parquet")
    pro = pd.read_parquet(D + f"probe2_process_factor_{fid}.parquet").rename(columns={"factor": "pf"})
    df = raw.merge(pro, on=["date", "instrument"])
    df["day"] = df["date"].dt.normalize()
    df = df.merge(ex[["day", "instrument"] + STYLE + IND], on=["day", "instrument"], how="left")
    days = sorted(df["date"].unique())[8:8 + 40]          # 取 40 个截面做快速比较
    sub = df[df["date"].isin(days)]
    rows = []
    for how in ["none", "mad3", "mad5", "madraw3", "madraw5", "sd3", "q1", "q2.5", "q5"]:
        for ind in [False, True]:
            vals = []
            for t, g in sub.groupby("date"):
                r = proc(g, how, ind)
                ok = np.isfinite(r) & g["pf"].notna().to_numpy()
                if ok.sum() > 10:
                    vals.append((np.corrcoef(r[ok], g["pf"].to_numpy()[ok])[0, 1], np.abs(r[ok] - g["pf"].to_numpy()[ok]).max()))
            v = np.array(vals)
            rows.append({"去极值": how, "含行业": ind, "与平台相关(均值)": v[:, 0].mean(), "最大绝对差(中位)": np.median(v[:, 1])})
    print(f"==== {fid}")
    print(pd.DataFrame(rows).sort_values("与平台相关(均值)", ascending=False).round(6).head(8).to_string(index=False))
