# c119（10-24）：14:30 截面的盘口与最后几分钟状态（1 分钟线直接取，因子库里可能没有）对"最后半小时漂移"的预测力。
import glob, numpy as np, pandas as pd
B = "/srv/media/jyc_scorer/bar1m_parquet"
cols = ["date", "instrument_id", "close", "volume", "amount", "high", "low", "ask_price1", "bid_price1", "ask_volume1", "bid_volume1",
        "ask_volume2", "bid_volume2", "ask_volume3", "bid_volume3", "ask_num_orders1", "bid_num_orders1", "ask_num_orders2", "bid_num_orders2"]
out = []
for y in (2023, 2024):
    for m in range(1, 13):
        p = pd.read_parquet(f"{B}/{y}{m:02d}.0.parquet", columns=cols)
        hm = p["date"].dt.hour * 100 + p["date"].dt.minute
        p = p[(hm >= 1420) & (hm <= 1430)].copy()
        p["hm"] = p["date"].dt.hour * 100 + p["date"].dt.minute
        p["day"] = p["date"].dt.normalize()
        p = p.sort_values(["instrument_id", "date"])
        g = p.groupby(["instrument_id", "day"])
        last = g.last()
        c1425 = p[p["hm"] <= 1425].groupby(["instrument_id", "day"])["close"].last()
        c1429 = p[p["hm"] <= 1429].groupby(["instrument_id", "day"])["close"].last()
        c1420 = p[p["hm"] <= 1420].groupby(["instrument_id", "day"])["close"].last()
        a1, b1 = last["ask_price1"], last["bid_price1"]
        av1, bv1 = last["ask_volume1"], last["bid_volume1"]
        mid = (a1 + b1) / 2
        ok = (a1 > b1) & (b1 > 0)
        d = pd.DataFrame(index=last.index)
        d["imb1"] = ((bv1 - av1) / (bv1 + av1)).where(ok & (bv1 + av1 > 0))
        dep_b = bv1 + last["bid_volume2"] + last["bid_volume3"]; dep_a = av1 + last["ask_volume2"] + last["ask_volume3"]
        d["imb3"] = ((dep_b - dep_a) / (dep_b + dep_a)).where(ok & (dep_a + dep_b > 0))
        micro = (a1 * bv1 + b1 * av1) / (bv1 + av1)
        d["micro"] = ((micro - mid) / mid).where(ok & (bv1 + av1 > 0))
        d["spr"] = ((a1 - b1) / mid).where(ok)
        d["last_vs_mid"] = ((last["close"] - mid) / mid).where(ok)
        nb, na = last["bid_num_orders1"], last["ask_num_orders1"]
        d["nord_imb"] = ((nb - na) / (nb + na)).where(nb + na > 0)
        d["size_b_a"] = np.log((bv1 / nb.where(nb > 0)) / (av1 / na.where(na > 0)))
        d["r1"] = last["close"] / c1429 - 1
        d["r5"] = last["close"] / c1425 - 1
        d["r10"] = last["close"] / c1420 - 1
        d["v10"] = g["volume"].sum()
        out.append(d.reset_index())
    print(y, flush=True)
d = pd.concat(out, ignore_index=True)
d["date"] = d["day"] + pd.Timedelta("14:30:00")
d["instrument"] = "ID" + d["instrument_id"].astype(int).astype(str).str.zfill(4)
f = pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c115_feats.parquet", columns=["date", "instrument", "drift", "auc", "y"])
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"]))
bf = pd.read_parquet(glob.glob("/home/user/jyc_scorer/jobs/1024-020416-70ad/input/*.parquet")[0]); bf["date"] = pd.to_datetime(bf["date"])
bf = bf[bf["date"].dt.strftime("%H%M") == "1430"].copy()
re_ = ~bf["instrument"].str.match(r"^ID\d{4}$"); bf.loc[re_, "instrument"] = bf.loc[re_, "instrument"].map(R2I)
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"]).rename(columns={"y": "ylab"})
lab = lab[lab["date"].dt.strftime("%H%M") == "1430"]
x = lab.merge(bf.dropna(subset=["instrument"]).rename(columns={"factor": "base"}), on=["date", "instrument"]).merge(f, on=["date", "instrument"], how="left").merge(d, on=["date", "instrument"], how="left")
codes, u = pd.factorize(x["date"]); G = len(u); yr = pd.DatetimeIndex(u).year
def cz(v):
    v = np.asarray(v, float); ok = np.isfinite(v)
    n = np.bincount(codes, ok, minlength=G); m = np.bincount(codes, np.where(ok, v, 0), minlength=G) / np.maximum(n, 1)
    return np.where(ok, v - m[codes], 0.0)
def rk(s): return (s.groupby(x["date"]).rank(pct=True) - 0.5).to_numpy()
def pc(a, b):
    a, b = cz(a), cz(b); r = np.bincount(codes, a * b, minlength=G) / np.sqrt(np.bincount(codes, a * a, minlength=G) * np.bincount(codes, b * b, minlength=G))
    return np.nanmean(r[yr == 2023]), np.nanmean(r[yr == 2024])
RD, RA, RY, RB = rk(x["drift"]), rk(x["auc"]), rk(x["ylab"]), rk(x["base"])
print("%-12s %-15s %-15s %-15s %s" % ("候选", "漂移 23/24", "竞价 23/24", "标签 23/24", "与底座相关"))
for c in ["imb1", "imb3", "micro", "spr", "last_vs_mid", "nord_imb", "size_b_a", "r1", "r5", "r10"]:
    r = rk(x[c])
    print("%-12s %+.3f/%+.3f  %+.3f/%+.3f  %+.3f/%+.3f  %+.3f" % (c, *pc(r, RD), *pc(r, RA), *pc(r, RY), pc(r, RB)[0]), flush=True)
