# -*- coding: utf-8 -*-
"""下午探索（10-19）：盘口深层字段的新根（第 2／3 档价格几乎没用过）。截面 t 只用 (t−30, t] 内的分钟（与平台根同一口径）。
  BK_THIN   盘口厚薄：第 3 档买卖价差 ÷ 第 1 档买卖价差（段内分钟均值），大 = 挂单价位稀疏
  BK_CONS   深度消耗率：段内成交量 ÷ 上一分钟第 1 档挂单量之和，大 = 主动交易吃盘口
  BK_DIMB   深层失衡：第 2、3 档 (买量 − 卖量) ÷ (买量 + 卖量)，段内均值（第 1 档失衡以前做过）
  *_20      过去 20 日（日均值）的均值，只用前几日，当日各截面相同
  *_DEV     当前段 − *_20
保存 vc_tmp/h1_BK_*.npy（原始值）。数据：macstudio 的原始 1 分钟 feather（61 个月），12 进程并行。"""
import glob
import sys
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id", "volume", "ask_price1", "ask_price3", "bid_price1", "bid_price3",
        "ask_volume1", "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3"]
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    ap1, bp1 = d["ask_price1"].to_numpy(float), d["bid_price1"].to_numpy(float)
    ap3, bp3 = d["ask_price3"].to_numpy(float), d["bid_price3"].to_numpy(float)
    ok = (bp1 > 0) & (ap1 > bp1) & (bp3 > 0) & (ap3 > bp3)
    thin = np.where(ok, (ap3 - bp3) / np.where(ok, ap1 - bp1, 1.0), np.nan)
    dep1 = (d["bid_volume1"].to_numpy(float) + d["ask_volume1"].to_numpy(float))
    dprev = pd.Series(dep1, index=d.index).groupby([d["instrument_id"], d["day"]], sort=False).shift(1).to_numpy()
    b23 = d["bid_volume2"].to_numpy(float) + d["bid_volume3"].to_numpy(float)
    a23 = d["ask_volume2"].to_numpy(float) + d["ask_volume3"].to_numpy(float)
    dimb = np.where(b23 + a23 > 0, (b23 - a23) / np.where(b23 + a23 > 0, b23 + a23, 1.0), np.nan)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "thin": thin, "vol": d["volume"].to_numpy(float), "dprev": np.where(dprev > 0, dprev, np.nan), "dimb": dimb})
    t = t[t["seg"] != ""]
    t["volc"] = np.where(np.isfinite(t["dprev"]), t["vol"], np.nan)
    a = t.groupby(["iid", "day", "seg"], sort=False).agg(BK_THIN=("thin", "mean"), v=("volc", "sum"), dp=("dprev", "sum"), BK_DIMB=("dimb", "mean"))
    a["BK_CONS"] = a["v"] / a["dp"].where(a["dp"] > 0)
    return a[["BK_THIN", "BK_CONS", "BK_DIMB"]].reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(12) as p:
        parts = p.map(one, files)
    A = pd.concat(parts, ignore_index=True)
    print("段行数", len(A), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    for nm in ("BK_THIN", "BK_CONS", "BK_DIMB"):
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        daym = Wd.mean(axis=1)
        slow = H1.roll_prev(daym, 20, 10)
        S = pd.DataFrame({s0: slow.to_numpy() for s0 in H1.CS}, index=Wd.index)
        for tag, fr in ((nm, Wd), (nm + "_20", S), (nm + "_DEV", Wd - S.to_numpy())):
            x = H1.align(c, fr)
            x[z930] = 0.0
            np.save(H1.OUT / ("h1_%s.npy" % tag), np.nan_to_num(x, nan=0.0))
            print("保存", tag, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
