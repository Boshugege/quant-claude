# c120（10-24）：14:30 截面"委托结构 × 竞价成交占比"的交互。假说：截面前的盘口压力在连续竞价里延续（漂移），在收盘竞价里回吐；
# 标签窗口 VWAP 含竞价成交，竞价成交占比大的股票标签由竞价跳动主导，小的由漂移主导，所以盘口信号对标签的符号应随（过去几天的）竞价占比翻转。
import os, glob, json, numpy as np, pandas as pd
B = "/srv/media/jyc_scorer/bar1m_parquet"; JOBS = "/home/user/jyc_scorer/jobs"
months = [(2022, 11), (2022, 12)] + [(y, m) for y in (2023, 2024) for m in range(1, 13)]
parts = []
for (y, m) in months:
    p = pd.read_parquet(f"{B}/{y}{m:02d}.0.parquet", columns=["date", "instrument_id", "volume"])
    hm = p["date"].dt.hour * 100 + p["date"].dt.minute
    p = p[(hm > 1430) & (hm <= 1500)].copy(); hm = hm[p.index]
    p["day"] = p["date"].dt.normalize(); p["va"] = np.where(hm == 1500, p["volume"], 0.0)
    g = p.groupby(["instrument_id", "day"]); parts.append(pd.DataFrame({"v": g["volume"].sum(), "va": g["va"].sum()}).reset_index())
d = pd.concat(parts).sort_values(["instrument_id", "day"]).reset_index(drop=True)
d["sh"] = d["va"] / d["v"].where(d["v"] > 0)
gi = d.groupby("instrument_id")["sh"]
for k in (1, 5, 20):
    d[f"sh{k}"] = gi.transform(lambda s: s.shift(1).rolling(k, min_periods=max(1, k // 2)).mean())
d = d[d["day"] >= "2023-01-01"].copy()
d["date"] = d["day"] + pd.Timedelta("14:30:00"); d["instrument"] = "ID" + d["instrument_id"].astype(int).astype(str).str.zfill(4)
d[["date", "instrument", "sh", "sh1", "sh5", "sh20"]].to_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c120_share.parquet", index=False)
f = pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c115_feats.parquet", columns=["date", "instrument", "drift", "auc"])
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"])
lab = lab[(lab["date"].dt.hour == 14) & (lab["date"].dt.minute == 30)]
x = lab.merge(f, on=["date", "instrument"], how="left").merge(d[["date", "instrument", "sh", "sh5", "sh20"]], on=["date", "instrument"], how="left")
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"]))
idx = {}
for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
    try: n = json.load(open(sp, encoding="utf-8")).get("name")
    except Exception: continue
    dd = os.path.dirname(sp)
    if n and os.path.exists(os.path.join(dd, "factor_raw.parquet")) and (n not in idx or dd > idx[n]): idx[n] = dd
def sig(nm):
    a = pd.read_parquet(os.path.join(idx[nm], "factor_raw.parquet"), columns=["date", "instrument", "factor"], filters=[("date", ">=", pd.Timestamp("2023-01-01"))])
    a["date"] = pd.to_datetime(a["date"]); a = a[(a["date"].dt.hour == 14) & (a["date"].dt.minute == 30)].copy()
    a["instrument"] = a["instrument"].map(lambda s: R2I.get(s, s))
    return x[["date", "instrument"]].merge(a.drop_duplicates(["date", "instrument"]), on=["date", "instrument"], how="left")["factor"].to_numpy()
def ic(v, t, mask=None):
    q = pd.DataFrame({"d": x["date"], "v": v, "t": x[t]})
    if mask is not None: q = q[mask]
    q = q.dropna(); c = q.groupby("d")[["v", "t"]].apply(lambda g: g["v"].corr(g["t"], method="spearman"))
    return c[c.index.year == 2023].mean(), c[c.index.year == 2024].mean()
ff = pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c115_feats.parquet", columns=["date", "instrument", "drift", "auc"]).sort_values(["instrument", "date"])
gg = ff.groupby("instrument")
for c in ("drift", "auc"):
    ff[c + "_sd"] = gg[c].transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
ff["rho"] = np.log(ff["auc_sd"] / ff["drift_sd"])
x = x.merge(ff[["date", "instrument", "rho"]], on=["date", "instrument"], how="left")
print("过去 20 天 竞价跳动波动 / 漂移波动（对数）中位 %.2f，四分位 %.2f～%.2f" % (x["rho"].median(), x["rho"].quantile(.25), x["rho"].quantile(.75)))
x["shr"] = x.groupby("date")["rho"].rank(pct=True)
print("竞价占比（窗口内）中位 %.3f，过去 5 天均值与当天的截面秩相关 %.3f" % (x["sh"].median(), x[["sh", "sh5"]].corr("spearman").iloc[0, 1]))
for nm in ["s30_td_s1", "s30_td_deep", "s21_sz_deep", "s20_ql_s1only", "s31_td_s1_sm0", "s35_of_l10_rel", "s20_ql_end"]:
    v = sig(nm)
    row = [nm]
    for lo, hi in ((0, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1.01)):
        mk = ((x["shr"] >= lo) & (x["shr"] < hi)).to_numpy()
        a = ic(v, "y", mk); row.append("%+.3f/%+.3f" % a)
    row.append("漂移 %+.3f/%+.3f 竞价 %+.3f/%+.3f" % (*ic(v, "drift"), *ic(v, "auc")))
    print("%-16s 标签 IC 按竞价相对漂移的波动三分位（低／中／高）：%s  %s  %s   %s" % tuple(row), flush=True)
