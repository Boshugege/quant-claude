# -*- coding: utf-8 -*-
"""反推平台 jyc_eval 的指标定义：平台因子值（probe_fac_*）+ 官方标签（probe_label_2024Q1），
逐项比较几种候选写法与平台自测结果（2024-01-02 至 03-01）。
用法：bash 工具/本地评分/remote.sh e2e/scripts/eval_def_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "e2e/src")
from feather_data import exposure_local  # noqa: E402

STYLE = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
PLAT = {"90": (0.0633322259124647, 0.884819179988961, 35.19718951107968, 0.7393099815983497, 0.6078728070175439),
        "98": (0.057305833536618424, 0.7827709715309709, 23.81336837332873, 0.5770154375349729, 0.5962218045112782),
        "100": (0.05685277850041038, 0.7569485088187401, 25.407016552771452, 0.5635654448725691, 0.6350000000000001),
        "102": (0.015037551107663851, 0.23463751354746704, 5.011670284032867, 0.18798810533839025, 0.10129699248120301),
        "104": (0.06588155783363826, 1.0881031767688831, 35.77950285632168, 0.9374708446856378, 0.6149122807017543)}
mp = pd.read_csv("数据/exposure/id_map.csv")
mp = mp[mp.ok]
lab = pd.read_parquet("数据/平台探测/probe_label_2024Q1.parquet")[["date", "instrument", "vwap_return"]]
lab = lab[(lab.date >= "2024-01-02") & (lab.date <= "2024-03-01 23:59:59")]
ex = pd.read_parquet("数据/exposure/exposure_2024.parquet", columns=["date", "instrument"] + STYLE)   # 真实代码，覆盖完整
ex["date"] = pd.to_datetime(ex["date"]).dt.normalize()


def winsor(x, how):
    if how == "none":
        return x
    if how == "mad3":
        med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) * 1.4826
        return np.clip(x, med - 3 * mad, med + 3 * mad)
    if how == "mad5":
        med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med)) * 1.4826
        return np.clip(x, med - 5 * mad, med + 5 * mad)
    if how == "sd3":
        mu, sd = np.nanmean(x), np.nanstd(x)
        return np.clip(x, mu - 3 * sd, mu + 3 * sd)
    if how == "q01":
        lo, hi = np.nanpercentile(x, [1, 99])
        return np.clip(x, lo, hi)
    if how == "q025":
        lo, hi = np.nanpercentile(x, [2.5, 97.5])
        return np.clip(x, lo, hi)


def process(df, how):
    out = []
    for t, g in df.groupby("date", sort=True):
        x = g["factor"].replace([np.inf, -np.inf], np.nan).to_numpy(np.float64)
        x = winsor(x, how)
        sd = np.nanstd(x)
        if not np.isfinite(sd) or sd == 0:
            out.append(pd.Series(np.nan, index=g.index)); continue
        z = (x - np.nanmean(x)) / sd
        B = np.nan_to_num(g[STYLE].to_numpy(np.float64))
        ok = np.isfinite(z)
        X = np.column_stack([np.ones(ok.sum()), B[ok]])
        coef, *_ = np.linalg.lstsq(X, z[ok], rcond=None)
        r = np.full(len(z), np.nan); r[ok] = z[ok] - X @ coef
        out.append(pd.Series(r, index=g.index))
    return pd.concat(out)


def label_resid(df):
    """标签也对 10 个风格逐截面回归取残差。"""
    out = []
    for t, g in df.groupby("date", sort=True):
        y = g["vwap_return"].to_numpy(np.float64)
        B = np.nan_to_num(g[STYLE].to_numpy(np.float64))
        ok = np.isfinite(y)
        X = np.column_stack([np.ones(ok.sum()), B[ok]])
        coef, *_ = np.linalg.lstsq(X, y[ok], rcond=None)
        r = np.full(len(y), np.nan); r[ok] = y[ok] - X @ coef
        out.append(pd.Series(r, index=g.index))
    return pd.concat(out)


def metrics(df, f, q=0.1, ic="spearman", turn="member_xday", n_ann=242 * 8, yic=None):
    d = df.assign(f=f, yic=df["vwap_return"] if yic is None else yic).dropna(subset=["f", "vwap_return"])
    ics, ls, ysd, legs_prev, tv, days = [], [], [], None, [], []
    for t, g in d.groupby("date", sort=True):
        if ic == "spearman":
            ics.append(g["f"].rank().corr(g["yic"].rank()))
        elif ic == "fillna0":
            ics.append(g["f"].rank().corr(g["yic"].rank()))
        else:
            ics.append(g["f"].corr(g["vwap_return"]))
        ysd.append(g["vwap_return"].std())
        n = max(int(len(g) * q), 1)
        o = g.sort_values("f")
        top, bot = set(o["instrument"].iloc[-n:]), set(o["instrument"].iloc[:n])
        ls.append(o["vwap_return"].iloc[-n:].mean() - o["vwap_return"].iloc[:n].mean())
        day = t.normalize()
        if legs_prev is not None and (turn == "member_xday" or day == legs_prev[2]):
            if turn == "long_only":
                tv.append(1 - len(top & legs_prev[0]) / n)
            else:
                tv.append((1 - len(top & legs_prev[0]) / n + 1 - len(bot & legs_prev[1]) / n) / 2)
        legs_prev = (top, bot, day)
    ics, ls, ysd = np.array(ics), np.array(ls), np.array(ysd)
    ir = lambda x: np.nanmean(x) / np.nanstd(x)  # noqa: E731
    hi = ysd > np.median(ysd)
    return (np.nanmean(ics), ir(ics), ls.mean() / ls.std() * np.sqrt(n_ann), min(ir(ics[hi]), ir(ics[~hi])), np.mean(tv))


rows = []
for fid, pv in PLAT.items():
    pf = pd.read_parquet(f"数据/平台探测/probe_fac_{fid}.parquet")
    df = pf.merge(lab, on=["date", "instrument"], how="inner")
    df["day"] = df["date"].dt.normalize()
    df = df.merge(ex.rename(columns={"date": "day"}), on=["day", "instrument"], how="left")
    rows.append({"id": fid, "variant": "平台", **dict(zip(["ic", "icir", "ls", "stress", "turn"], pv))})
    for how in ["none", "mad3", "mad5", "sd3", "q01", "q025"]:
        f = process(df, how)
        for icm in ["spearman", "pearson"]:
            r = metrics(df, f, ic=icm)
            rows.append({"id": fid, "variant": f"{how}/{icm}", **dict(zip(["ic", "icir", "ls", "stress", "turn"], r))})
    f = process(df, "mad3")
    yr = label_resid(df)
    r = metrics(df, f, turn="same_day", yic=yr)
    rows.append({"id": fid, "variant": "标签也剔除风格", **dict(zip(["ic", "icir", "ls", "stress", "turn"], r))})
    yw = df.groupby("date")["vwap_return"].transform(lambda x: x.clip(*np.nanpercentile(x, [1, 99])))
    r = metrics(df, f, turn="same_day", yic=yw, ic="pearson")
    rows.append({"id": fid, "variant": "Pearson(残差因子, 去极值标签)", **dict(zip(["ic", "icir", "ls", "stress", "turn"], r))})
    ff = df.assign(f=f).groupby("date")["f"].rank(pct=True)
    r = metrics(df, ff, turn="same_day", ic="pearson")
    rows.append({"id": fid, "variant": "Pearson(因子排名, 原始标签)", **dict(zip(["ic", "icir", "ls", "stress", "turn"], r))})
    df0 = df.copy(); df0["factor"] = df0["factor"].where(df0["factor"] != 0)
    r = metrics(df0, process(df0, "mad3"), turn="same_day")
    rows.append({"id": fid, "variant": "0 视为缺失", **dict(zip(["ic", "icir", "ls", "stress", "turn"], r))})
    for q, turn in [(0.1, "same_day"), (0.2, "member_xday"), (0.1, "long_only"), (0.2, "same_day")]:
        r = metrics(df, f, q=q, turn=turn)
        rows.append({"id": fid, "variant": f"mad3/换手 q={q} {turn}", **dict(zip(["ic", "icir", "ls", "stress", "turn"], r))})
    print(fid, "done", flush=True)
t = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(t[t.variant.isin(["平台", "mad3/换手 q=0.1 same_day", "标签也剔除风格", "Pearson(残差因子, 去极值标签)", "Pearson(因子排名, 原始标签)", "0 视为缺失"])].round(4).to_string(index=False))
print("暴露缺失率", float(df[STYLE].isna().any(axis=1).mean()))
# 每种写法相对平台的平均绝对误差
plat = t[t.variant == "平台"].set_index("id")
err = []
for v, g in t[t.variant != "平台"].groupby("variant"):
    g = g.set_index("id")
    err.append({"variant": v, **{c: float((g[c] - plat[c]).abs().mean()) for c in ["ic", "icir", "ls", "stress", "turn"]}})
print(pd.DataFrame(err).sort_values("icir").round(5).to_string(index=False))
