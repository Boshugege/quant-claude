# c126（10-24）：每个截面"当天至今收益"（截面收盘 ÷ 复权昨收 − 1）与风格暴露日变化对当天收益的斜率，
# 供 c105j 的 RESID=intra：暴露 = 前一交易日暴露 + 斜率 × 当天至今收益（只用截面时刻及以前的数据）。
import numpy as np, pandas as pd
B = "/srv/media/jyc_scorer/bar1m_parquet"; E = "/srv/media/jyc_scorer/incoming/e2eslot"
SL = [1000, 1030, 1100, 1130, 1330, 1400, 1430]
out = []
months = [(2022, 12)] + [(y, m) for y in (2023, 2024) for m in range(1, 13)]
for (y, m) in months:
    p = pd.read_parquet(f"{B}/{y}{m:02d}.0.parquet", columns=["date", "instrument_id", "adjust_factor", "close"])
    p = p[p["close"] > 0].sort_values(["instrument_id", "date"])
    p["day"] = p["date"].dt.normalize(); hm = p["date"].dt.hour * 100 + p["date"].dt.minute
    last = p.groupby(["instrument_id", "day"]).agg(cday=("close", "last"), adj=("adjust_factor", "last")).reset_index()
    rows = []
    for s in SL:
        c = p[hm <= s].groupby(["instrument_id", "day"])["close"].last().rename("c").reset_index()
        c["slot"] = s; rows.append(c)
    c = pd.concat(rows)
    out.append((last, c))
    print(y, m, flush=True)
last = pd.concat([o[0] for o in out]).drop_duplicates(["instrument_id", "day"]).sort_values(["instrument_id", "day"])
g = last.groupby("instrument_id")
last["pc"] = g["cday"].shift(1) * g["adj"].shift(1) / last["adj"]
c = pd.concat([o[1] for o in out]).merge(last[["instrument_id", "day", "pc"]], on=["instrument_id", "day"], how="left")
c["r"] = c["c"] / c["pc"] - 1
c = c[c["day"] >= "2023-01-01"].copy()
c["date"] = c["day"] + pd.to_timedelta((c["slot"] // 100).astype(str) + ":" + (c["slot"] % 100).astype(str).str.zfill(2) + ":00")
c["instrument"] = "ID" + c["instrument_id"].astype(int).astype(str).str.zfill(4)
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
I2R = dict(zip(_m["local_id"], _m["instrument"]))
c["instrument"] = c["instrument"].map(lambda s: I2R.get(s, s))
c[["date", "instrument", "r"]].replace([np.inf, -np.inf], np.nan).to_parquet(f"{E}/c126_rsofar.parquet", index=False)
print("当天至今收益", len(c), "缺失 %.3f" % c["r"].isna().mean())
# 斜率：每天截面上 Δ暴露 对 当天收益 的回归系数，取 2020–2022 的均值（避免用评估期）
S = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
e = pd.concat([pd.read_parquet(f"/home/user/jyc_scorer/data/exposure/exposure_{y}.parquet", columns=["date", "instrument", "ret"] + S) for y in (2020, 2021, 2022)])
e["day"] = pd.to_datetime(e["date"]).dt.normalize(); e = e.sort_values(["instrument", "day"])
gg = e.groupby("instrument")
sl = {}
for k in S:
    d = e[k] - gg[k].shift(1)
    q = pd.DataFrame({"day": e["day"], "d": d, "r": e["ret"]}).dropna()
    b = q.groupby("day").apply(lambda t: np.cov(t["d"], t["r"])[0, 1] / np.var(t["r"], ddof=1) if len(t) > 50 else np.nan, include_groups=False)
    sl[k] = float(b.median())
print("斜率（Δ暴露 / 当天收益，2020–22 每日中位）:", {k: round(v, 2) for k, v in sl.items()})
import json; json.dump(sl, open(f"{E}/c126_slopes.json", "w"))
