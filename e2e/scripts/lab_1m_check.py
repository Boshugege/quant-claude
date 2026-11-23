# -*- coding: utf-8 -*-
"""用 1 分钟数据重建官方 vwap_return 的几种窗口/清洗口径，与官方表逐行比较（2024-01 前两周）。
用法：bash e2e/scripts/remote.sh e2e/scripts/lab_1m_check.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "e2e/src")
from feather_data import read_range  # noqa: E402

mp = pd.read_csv("数据/exposure/id_map.csv")
mp = mp[mp.ok]
lab = pd.read_parquet("数据/平台探测/probe_label_2024Q1.parquet")
lab = lab[lab.date < "2024-01-13"].merge(mp[["instrument", "local_id"]], on="instrument")
ids = sorted(lab.local_id.unique())
b = read_range("2024-01-02 00:00:00", "2024-01-12 23:59:59", ids)[["date", "instrument", "close", "high", "low", "volume", "amount"]]
b["day"] = b.date.dt.normalize()
hm = b.date.dt.hour * 100 + b.date.dt.minute
# 每分钟归属的截面：(t, t+30]；11:30 截面对应 13:00–13:30
edges = [(930, 1000, 930), (1000, 1030, 1000), (1030, 1100, 1030), (1100, 1130, 1100), (1300, 1330, 1130),
         (1330, 1400, 1330), (1400, 1430, 1400), (1430, 1500, 1430)]
sec = pd.Series(-1, index=b.index)
for lo, hi, s in edges:
    sec[(hm > lo) & (hm <= hi)] = s
b["sec"] = sec
b["hm"] = hm
b = b[b.sec >= 0].sort_values(["instrument", "date"])
mv = b.amount / b.volume.where(b.volume > 0)
b["clean"] = mv.notna() & (mv >= b.low * 0.98) & (mv <= b.high * 1.02)
lab["sec"] = lab.date.dt.hour * 100 + lab.date.dt.minute
lab["day"] = lab.date.dt.normalize()
res = []
for name, mask in [("全部分钟", b.volume >= 0), ("分钟清洗", b.clean), ("去掉开盘第一分钟", ~((b.hm == 931))),
                   ("去掉每窗第一分钟", b.groupby(["instrument", "day", "sec"]).cumcount() > 0)]:
    w = b[mask]
    agg = w.groupby(["instrument", "day", "sec"]).agg(a=("amount", "sum"), v=("volume", "sum"))
    last = b.groupby(["instrument", "day", "sec"]).close.last().rename("c")
    agg = agg.join(last).reset_index().rename(columns={"instrument": "local_id"})
    m = lab.merge(agg, on=["local_id", "day", "sec"])
    y = m.c / (m.a / m.v) - 1
    ex = (y - m.vwap_return).abs() < 1e-5
    res.append({"口径": name, "总体一致率": round(float(ex.mean()), 4), **{str(s): round(float(ex[m.sec == s].mean()), 3) for s in sorted(m.sec.unique())}})
print(pd.DataFrame(res).to_string(index=False))

# “全部分钟”口径的细节：成交额、成交量、收盘价与官方的差异，以及截面秩相关
w = b
agg = w.groupby(["instrument", "day", "sec"]).agg(a=("amount", "sum"), v=("volume", "sum"), c=("close", "last"), n=("close", "size")).reset_index().rename(columns={"instrument": "local_id"})
m = lab.merge(agg, on=["local_id", "day", "sec"])
m["y"] = m.c / (m.a / m.v) - 1
m["ra"] = m.a / m.amount - 1
m["rv"] = m.v / m.volume - 1
m["ce"] = (m.c - m.end_price).abs() < 1e-4
out = m.groupby("sec").agg(额相对差中位=("ra", lambda x: x.abs().median()), 额完全一致=("ra", lambda x: (x.abs() < 1e-9).mean()),
                           量完全一致=("rv", lambda x: (x.abs() < 1e-9).mean()), 收盘一致=("ce", "mean"), 分钟数=("n", "median"))
out["秩相关"] = m.groupby("sec").apply(lambda g: g.groupby("day").apply(lambda h: h.y.rank().corr(h.vwap_return.rank())).mean())
print(out.round(5).to_string())
x = m[(m.sec == 930) & (m.ra.abs() > 1e-6)].head(5)
print(x[["day", "local_id", "amount", "a", "volume", "v", "end_price", "c"]].to_string())

# 差额 = 本地 - 官方，看它是否恰好等于窗口里（或窗口边界外）某一分钟的成交额；收盘价是否等于某一分钟的收盘价
bb = read_range("2024-01-02 00:00:00", "2024-01-12 23:59:59", ids)[["date", "instrument", "close", "amount"]]
bb["day"] = bb.date.dt.normalize(); bb["hm"] = bb.date.dt.hour * 100 + bb.date.dt.minute
piv_a = bb.pivot_table(index=["instrument", "day"], columns="hm", values="amount", aggfunc="sum")
piv_c = bb.pivot_table(index=["instrument", "day"], columns="hm", values="close", aggfunc="last")
m["dd"] = m.a - m.amount
for s, cand in [(930, [925, 930, 931, 932, 1000]), (1130, [1130, 1300, 1301, 1302, 1330]), (1000, [1000, 1001, 1030]), (1330, [1330, 1331, 1400])]:
    mm = m[(m.sec == s) & (m["dd"].abs() > 1)]
    key = list(zip(mm.local_id, mm.day))
    r = {}
    for k in cand:
        if k in piv_a.columns:
            av = piv_a[k].reindex(key).to_numpy()
            r[f"差额=分钟{k}额"] = round(float((np.abs(mm["dd"].to_numpy() - av) < 1).mean()), 3)
    print("截面", s, "差额非零行", len(mm), r)
    mc = m[(m.sec == s) & ~m.ce]
    key = list(zip(mc.local_id, mc.day))
    r = {}
    for k in sorted(piv_c.columns):
        if (s == 930 and 955 <= k <= 1005) or (s == 1130 and 1325 <= k <= 1335) or (s == 1000 and 1025 <= k <= 1035) or (s == 1330 and 1355 <= k <= 1405):
            cv = piv_c[k].reindex(key).to_numpy()
            r[k] = round(float((np.abs(mc.end_price.to_numpy() - cv) < 1e-4).mean()), 3)
    print("   收盘价不一致行", len(mc), "end_price 等于某分钟收盘的比例", r)
