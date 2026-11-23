# -*- coding: utf-8 -*-
"""平台探测 02 分析：平台处理后的因子值（process_factor）与本地处理比较，并在平台处理值上反推 IC / 多空夏普的定义。
用法（本机即可）：python3 e2e/scripts/probe2_analyze.py
"""
import numpy as np
import pandas as pd

D = "数据/平台探测/"
STYLE = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
PLAT = {"90": (0.0633322259124647, 0.884819179988961, 35.19718951107968, 0.7393099815983497, 0.6078728070175439),
        "104": (0.06588155783363826, 1.0881031767688831, 35.77950285632168, 0.9374708446856378, 0.6149122807017543)}
lab = pd.read_parquet(D + "probe_label_2024Q1.parquet")[["date", "instrument", "vwap_return"]]
ex = pd.read_parquet("数据/exposure/exposure_2024.parquet", columns=["date", "instrument"] + STYLE)
ex["day"] = pd.to_datetime(ex["date"]).dt.normalize()
ex = ex.drop(columns="date")
pd.set_option("display.width", 250)


def local_proc(df):
    out = pd.Series(np.nan, index=df.index)
    for t, g in df.groupby("date"):
        x = g["factor"].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
        if x.std() == 0:
            continue
        z = (x - x.mean()) / x.std()
        B = g[STYLE].to_numpy(float)
        X = np.column_stack([np.ones(len(z)), B])
        c, *_ = np.linalg.lstsq(X, z, rcond=None)
        out[g.index] = z - X @ c
    return out


def metrics(df, f, y="vwap_return", ic="spearman", q=0.1, ls_mode="rank", min_n=1):
    d = df.assign(f=f).dropna(subset=["f", y])
    ics, ls, sd = [], [], []
    for t, g in d.groupby("date", sort=True):
        if len(g) < min_n:
            continue
        ics.append(g["f"].rank().corr(g[y].rank()) if ic == "spearman" else g["f"].corr(g[y]))
        sd.append(g[y].std())
        if ls_mode == "rank":
            n = max(int(len(g) * q), 1)
            o = g.sort_values("f")
            ls.append(o[y].iloc[-n:].mean() - o[y].iloc[:n].mean())
        else:  # qcut 分 10 组，最高组减最低组
            grp = pd.qcut(g["f"].rank(method="first"), 10, labels=False)
            ls.append(g[y][grp == 9].mean() - g[y][grp == 0].mean())
    ics, ls, sd = np.array(ics), np.array(ls), np.array(sd)
    ir = lambda x: np.nanmean(x) / np.nanstd(x)  # noqa: E731
    hi = sd > np.median(sd)
    return np.nanmean(ics), ir(ics), ls.mean() / ls.std() * np.sqrt(242 * 8), min(ir(ics[hi]), ir(ics[~hi])), len(ics), ics


for fid, pv in PLAT.items():
    raw = pd.read_parquet(D + f"probe2_raw_factor_{fid}.parquet")
    pro = pd.read_parquet(D + f"probe2_process_factor_{fid}.parquet").rename(columns={"factor": "pf"})
    df = raw.merge(pro, on=["date", "instrument"]).merge(lab, on=["date", "instrument"], how="left")
    df["day"] = df["date"].dt.normalize()
    df = df.merge(ex, on=["day", "instrument"], how="left").reset_index(drop=True)
    df["lf"] = local_proc(df)
    ok = df[["pf", "lf"]].notna().all(1)
    print(f"\n######## {fid}  平台处理值缺失率 {df.pf.isna().mean():.4f}  标签缺失率 {df.vwap_return.isna().mean():.4f}")
    print("  平台处理值 vs 本地处理值：相关", round(float(df.loc[ok, "pf"].corr(df.loc[ok, "lf"])), 6),
          " 最大差", round(float((df.loc[ok, "pf"] - df.loc[ok, "lf"]).abs().max()), 6))
    st = df.groupby("date")["pf"].agg(["mean", "std", "min", "max", "count"])
    print("  平台处理值逐截面统计（中位）:", st.median().round(4).to_dict())
    rows = [("平台", *pv[:4], None)]
    for name, f, kw in [("平台处理值 Spearman", df.pf, {}), ("平台处理值 Pearson", df.pf, {"ic": "pearson"}),
                        ("平台处理值 qcut10 多空", df.pf, {"ls_mode": "qcut"}), ("本地处理值 Spearman", df.lf, {})]:
        m = metrics(df, f, **kw)
        rows.append((name, *m[:4], m[4]))
    print(pd.DataFrame(rows, columns=["口径", "ic", "icir", "ls", "stress", "截面数"]).round(4).to_string(index=False))
    # 平台截面数：IC 均值若按 304 个截面（含 NaN 当 0）计算
    m = metrics(df, df.pf)
    ics = m[5]
    print("  若把缺失截面的 IC 记为 0、按 304 个截面平均:", round(float(np.nansum(ics) / 304), 4),
          " 有效截面", len(ics))
