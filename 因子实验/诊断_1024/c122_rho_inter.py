# c122（10-24）：14:30 截面"委托结构 × 竞价相对漂移的波动"交互项，交评分机（values），供 c105i 的 ADD=任务@1430:权重 使用。
# 交互 = z(信号) × (rank(ρ) − 0.5)，ρ = ln(过去 20 天竞价跳动波动 ÷ 过去 20 天最后半小时漂移波动)；只在 14:30 有值，其它截面 0。
import os, glob, json, subprocess, numpy as np, pandas as pd
JOBS = "/home/user/jyc_scorer/jobs"; E = "/srv/media/jyc_scorer/incoming/e2eslot"
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"])); I2R = dict(zip(_m["local_id"], _m["instrument"]))
ff = pd.read_parquet(f"{E}/c115_feats.parquet", columns=["date", "instrument", "drift", "auc"]).sort_values(["instrument", "date"])
gg = ff.groupby("instrument")
for c in ("drift", "auc"):
    ff[c + "_sd"] = gg[c].transform(lambda s: s.shift(1).rolling(20, min_periods=10).std())
ff["rho"] = np.log(ff["auc_sd"] / ff["drift_sd"])
ff["rr"] = ff.groupby("date")["rho"].rank(pct=True) - 0.5
idx = {}
for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
    try: n = json.load(open(sp, encoding="utf-8")).get("name")
    except Exception: continue
    dd = os.path.dirname(sp)
    if n and os.path.exists(os.path.join(dd, "factor_raw.parquet")) and (n not in idx or dd > idx[n]): idx[n] = dd
keys = pd.concat([pd.read_parquet(f"{E}/e2e_R22a_{y}.parquet", columns=["date", "instrument"]) for y in (2023, 2024)])
keys["date"] = pd.to_datetime(keys["date"])
for nm in ["s30_td_s1", "s20_ql_s1only", "s30_td_deep"]:
    a = pd.read_parquet(os.path.join(idx[nm], "factor_raw.parquet"), columns=["date", "instrument", "factor"], filters=[("date", ">=", pd.Timestamp("2023-01-01"))])
    a["date"] = pd.to_datetime(a["date"]); a = a[(a["date"].dt.hour == 14) & (a["date"].dt.minute == 30)].copy()
    a["instrument"] = a["instrument"].map(lambda s: R2I.get(s, s))
    a = a.drop_duplicates(["date", "instrument"]).merge(ff[["date", "instrument", "rr"]], on=["date", "instrument"], how="inner")
    g = a.groupby("date")["factor"]
    a["z"] = ((a["factor"] - g.transform("mean")) / g.transform("std")).fillna(0.0)
    a["v"] = a["z"] * a["rr"].fillna(0.0)
    v = keys.merge(a[["date", "instrument", "v"]], on=["date", "instrument"], how="left").rename(columns={"v": "factor"})
    v["factor"] = v["factor"].fillna(0.0)
    v["instrument"] = v["instrument"].map(lambda s: I2R.get(s, s))
    out = f"{E}/c122_{nm}_xrho.parquet"; v.to_parquet(out, index=False)
    r = subprocess.run(["curl", "-s", "-F", f"files=@{out}", "-F", "kind=values", "-F", "start=2023-01", "-F", "end=2024-12",
                        "-F", f"name=c122_{nm}_xrho", "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print(nm, "非零", int((v["factor"] != 0).sum()), r.stdout[:90], flush=True)
