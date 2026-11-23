# c115（10-24）：14:30 截面"最后半小时连续竞价漂移"的预测因子实验室。
# 端到端在 14:30 的优势全部来自漂移部分（14:57 收盘 / 14:31–14:57 VWAP − 1：端到端 IC 0.135，1663 0.090）；收盘竞价跳动部分因子反而更好（0.052 对 0.030）。
# 从 1 分钟线直接造候选（只用 14:30 及以前），算对漂移／竞价跳动／完整标签的 IC，以及加到 1663 不加缓冲带输出（14:30）上的增量。
import sys, numpy as np, pandas as pd
B = "/srv/media/jyc_scorer/bar1m_parquet"
months = [(2022, 11), (2022, 12)] + [(y, m) for y in (2023, 2024) for m in range(1, 13)]
cols = ["date", "instrument_id", "adjust_factor", "close", "volume", "amount", "high", "low"]

def last_before(p, hm, h, name):
    q = p[hm <= h]
    return q.groupby(["instrument_id", "day"])["close"].last().rename(name)

out = []
for (y, m) in months:
    p = pd.read_parquet(f"{B}/{y}{m:02d}.0.parquet", columns=cols)
    p = p[p["close"] > 0].sort_values(["instrument_id", "date"])
    hm = (p["date"].dt.hour * 100 + p["date"].dt.minute).to_numpy()
    p["day"] = p["date"].dt.normalize()
    vw = p["amount"] / p["volume"].where(p["volume"] > 0)
    ok = vw.notna() & (vw >= p["low"] * 0.98) & (vw <= p["high"] * 1.02)
    p["amt"] = p["amount"].where(ok, 0.0); p["vol"] = p["volume"].where(ok, 0.0)
    key = ["instrument_id", "day"]
    def vwap(mask, name):
        g = p[mask].groupby(key)
        return (g["amt"].sum() / g["vol"].sum().where(lambda s: s > 0)).rename(name)
    def vsum(mask, name):
        return p[mask].groupby(key)["vol"].sum().rename(name)
    feats = [
        vwap((hm > 1430) & (hm <= 1457), "vw_c"), vwap((hm > 1430) & (hm <= 1500), "vw_all"),
        vwap((hm > 1400) & (hm <= 1430), "vw_1400"), vwap((hm >= 930) & (hm <= 1430), "vw_day"),
        vwap((hm > 1330) & (hm <= 1400), "vw_1330"),
        vsum((hm > 1400) & (hm <= 1430), "v_1400"), vsum((hm >= 930) & (hm <= 1430), "v_day"),
        last_before(p, hm, 1457, "c1457"), last_before(p, hm, 1500, "c1500"), last_before(p, hm, 1430, "c1430"),
        last_before(p, hm, 1400, "c1400"), last_before(p, hm, 1330, "c1330"), last_before(p, hm, 1130, "c1130"),
        last_before(p, hm, 1000, "c1000"),
        p[(hm >= 930) & (hm <= 1430)].groupby(key)["high"].max().rename("hi"),
        p[(hm >= 930) & (hm <= 1430)].groupby(key)["low"].min().rename("lo"),
        p.groupby(key)["close"].last().rename("cday"), p.groupby(key)["adjust_factor"].last().rename("adj"),
    ]
    d = pd.concat(feats, axis=1).reset_index()
    out.append(d)
    print(y, m, len(d), flush=True)
d = pd.concat(out, ignore_index=True).sort_values(["instrument_id", "day"]).reset_index(drop=True)
d = d.replace([np.inf, -np.inf], np.nan)
g = d.groupby("instrument_id")
d["pc"] = g["cday"].shift(1) * g["adj"].shift(1) / d["adj"]           # 复权后的昨收
d["y"] = d["c1500"] / d["vw_all"] - 1
d["drift"] = d["c1457"] / d["vw_c"] - 1
d["auc"] = d["c1500"] / d["c1457"] - 1
# —— 当天 14:30 及以前的信息
F = {}
F["seg"] = d["c1430"] / d["c1400"] - 1                                   # 最后一段收益
F["cv1400"] = d["c1430"] / d["vw_1400"] - 1                              # 上一窗口的"末价 / VWAP"（与标签同形）
F["cv1330"] = d["c1400"] / d["vw_1330"] - 1                              # 再上一窗口
F["pos_day"] = d["c1430"] / d["vw_day"] - 1                              # 相对当天 VWAP 的位置
F["r_day"] = d["c1430"] / d["pc"] - 1                                    # 当天至今收益
F["r_am30"] = d["c1000"] / d["pc"] - 1                                   # 开盘 30 分钟（日内动量文献的第一段）
F["r_pm"] = d["c1430"] / d["c1130"] - 1                                  # 午后至今
F["hl_pos"] = (d["c1430"] - d["lo"]) / (d["hi"] - d["lo"]).where(lambda s: s > 0)
F["v_last"] = d["v_1400"] / d["v_day"].where(lambda s: s > 0)            # 最后一段成交量占当天比例
F["seg_x_vlast"] = F["seg"] * F["v_last"]
# —— 过去几天（严格早于当天）
for nm, col in [("drift", "drift"), ("auc", "auc"), ("y", "y")]:
    s = g[col].shift(1)
    F[f"{nm}_1"] = s
    F[f"{nm}_5"] = s.groupby(d["instrument_id"]).transform(lambda v: v.rolling(5, min_periods=3).mean())
    F[f"{nm}_20"] = s.groupby(d["instrument_id"]).transform(lambda v: v.rolling(20, min_periods=10).mean())
F["cvx_20"] = (g["drift"].shift(1)).groupby(d["instrument_id"]).transform(lambda v: v.rolling(20, min_periods=10).apply(lambda a: np.mean(np.sign(a)), raw=True))
for k, v in F.items():
    d["f_" + k] = v
d = d[d["day"] >= "2023-01-01"].copy()
d["date"] = d["day"] + pd.Timedelta("14:30:00")
d["instrument"] = "ID" + d["instrument_id"].astype(int).astype(str).str.zfill(4)
d = d.replace([np.inf, -np.inf], np.nan)
d.to_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c115_feats.parquet", index=False)

# —— 底座：1663 不加缓冲带（14:30），评估股票 = 标签缓存里有的
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str)
_m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"]))
import glob
bf = pd.read_parquet(glob.glob("/home/user/jyc_scorer/jobs/1024-020416-70ad/input/*.parquet")[0])
bf["date"] = pd.to_datetime(bf["date"])
bf = bf[bf["date"].dt.strftime("%H%M") == "1430"].copy()
re_ = ~bf["instrument"].str.match(r"^ID\d{4}$")
bf.loc[re_, "instrument"] = bf.loc[re_, "instrument"].map(R2I)
bf = bf.dropna(subset=["instrument"]).rename(columns={"factor": "base"})
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"]).rename(columns={"y": "ylab"})
lab = lab[lab["date"].dt.strftime("%H%M") == "1430"]
x = lab.merge(bf, on=["date", "instrument"], how="inner").merge(d, on=["date", "instrument"], how="left")
print("评估行数", len(x), "标签缓存 vs 重建 y 的秩相关 %.3f" % x[["ylab", "y"]].corr("spearman").iloc[0, 1], flush=True)

def zs(s):
    gg = s.groupby(x["date"])
    return ((s - gg.transform("mean")) / gg.transform("std").replace(0, np.nan)).fillna(0.0)
x["zb"] = zs(x["base"])
rk = lambda s: s.groupby(x["date"]).rank(pct=True) - 0.5
x["rl"] = rk(x["ylab"])
def ic_by_year(f, t):
    q = x[["date", f, t]].dropna()
    ic = q.groupby("date")[[f, t]].apply(lambda g: g[f].corr(g[t], method="spearman"))
    return ic[ic.index.year == 2023].mean(), ic[ic.index.year == 2024].mean()
def comb_ic(v):
    q = pd.DataFrame({"date": x["date"], "v": v, "r": x["rl"]}).dropna()
    ic = q.groupby("date")[["v", "r"]].apply(lambda g: g["v"].corr(g["r"]))
    return ic[ic.index.year == 2023].mean(), ic[ic.index.year == 2024].mean()
b23, b24 = comb_ic(x["zb"])
print("底座 14:30（Pearson 对标签秩）2023 %.4f 2024 %.4f" % (b23, b24))
print("%-14s %-15s %-15s %-15s %s" % ("候选", "IC 漂移 23/24", "IC 竞价 23/24", "IC 标签 23/24", "加到底座 Δ(23/24)：w=+0.1 +0.2 −0.1 −0.2"))
for k in F:
    f = "f_" + k
    a = ic_by_year(f, "drift"); b = ic_by_year(f, "auc"); c = ic_by_year(f, "ylab")
    zf = zs(x[f])
    ds = []
    for w in (0.1, 0.2, -0.1, -0.2):
        c23, c24 = comb_ic(x["zb"] + w * zf)
        ds.append("%+.4f/%+.4f" % (c23 - b23, c24 - b24))
    print("%-14s %+.3f/%+.3f  %+.3f/%+.3f  %+.3f/%+.3f  %s" % (k, a[0], a[1], b[0], b[1], c[0], c[1], "  ".join(ds)), flush=True)
