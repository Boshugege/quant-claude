# -*- coding: utf-8 -*-
"""T12 的特征（10-23 晚，用户："不要拘泥于小的微调"）：把这几天找到的新信息直接交给 1617 的树，让树自己学交互（例如 14:30 方向相反）。
  OF10   截面前 10 分钟主动买卖失衡 (主动买 − 主动卖) / (主动买 + 主动卖)，主动方向按分钟收盘在买一卖一之间的位置（同 s35）
  CLV10  截面价在最后 10 分钟高低区间中的位置（2·收 − 高 − 低）/（高 − 低）
  AUC5   过去 5 个交易日收盘集合竞价收益 ln(15:00 价 ÷ 14:56 收盘) 的均值（只用之前的交易日，同 s37）
  AUC20  过去 20 个交易日的均值
截面 t 只用 t 及之前的数据。保存 vc_tmp/h1_<名>.npy（原始值）。写法同 m26。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
LAST = {"1000": 951, "1030": 1021, "1100": 1051, "1130": 1121, "1330": 1321, "1400": 1351, "1430": 1421}
COLS = ["date", "instrument_id", "high", "low", "close", "volume", "bid_price1", "ask_price1"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    last = np.zeros(len(d), dtype=bool)
    for a, b, s in EDGES:
        k = (hm >= a) & (hm <= b)
        seg[k] = s
        last |= k & (hm >= LAST[s])
    c, v, b1, a1 = (d[x].to_numpy(float) for x in ("close", "volume", "bid_price1", "ask_price1"))
    ok = (b1 > 0) & (a1 > b1) & (c > 0)
    tpos = np.where(ok, np.clip((c - b1) / np.where(ok, a1 - b1, 1.0), 0.0, 1.0), 0.5)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg, "hm": hm,
                      "vb": np.where(last, v * tpos, 0.0), "vs": np.where(last, v * (1 - tpos), 0.0),
                      "hi": np.where(last & (d["high"].to_numpy(float) > 0), d["high"].to_numpy(float), np.nan),
                      "lo": np.where(last & (d["low"].to_numpy(float) > 0), d["low"].to_numpy(float), np.nan), "c": c})
    s = t[t["seg"] != ""]
    g = s.groupby(["iid", "day", "seg"], sort=False).agg(vb=("vb", "sum"), vs=("vs", "sum"), hi=("hi", "max"), lo=("lo", "min"), c=("c", "last"))
    out = pd.DataFrame(index=g.index)
    out["OF10"] = (g["vb"] - g["vs"]) / (g["vb"] + g["vs"]).where(g["vb"] + g["vs"] > 0)
    out["CLV10"] = ((2 * g["c"] - g["hi"] - g["lo"]) / (g["hi"] - g["lo"]).where(g["hi"] > g["lo"])).clip(-1, 1)
    au = t[(t["hm"] == 1456) | (t["hm"] == 1500)].pivot_table(index=["iid", "day"], columns="hm", values="c", aggfunc="last")
    au = np.log(au[1500].where(au[1500] > 0) / au[1456].where(au[1456] > 0)).rename("ra").reset_index() if {1456, 1500} <= set(au.columns) else pd.DataFrame(columns=["iid", "day", "ra"])
    return out.replace([np.inf, -np.inf], np.nan).reset_index(), au


def main():
    with Pool(10) as p:
        res = p.map(one, sorted(glob.glob(SRC)))
    A = pd.concat([r[0] for r in res], ignore_index=True)
    U = pd.concat([r[1] for r in res], ignore_index=True).sort_values(["iid", "day"])
    gi = U.groupby("iid")["ra"]
    U["AUC5"] = gi.transform(lambda s: s.shift(1).rolling(5, min_periods=3).mean())
    U["AUC20"] = gi.transform(lambda s: s.shift(1).rolling(20, min_periods=10).mean())
    for nm in ("AUC5", "AUC20"):                                                 # 当天各截面同一个值（只用之前的交易日）
        A = A.merge(U[["iid", "day", nm]], on=["iid", "day"], how="left")
    print("段行数", len(A), A[["OF10", "CLV10", "AUC5", "AUC20"]].notna().mean().round(3).to_dict(), flush=True)
    c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
    for nm in ("OF10", "CLV10", "AUC5", "AUC20"):
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        x = H1.align(c, Wd)
        x = np.where(np.isfinite(x) & (x == 0), 1e-9, x); x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
