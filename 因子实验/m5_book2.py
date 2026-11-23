# -*- coding: utf-8 -*-
"""下午探索第二批（10-19）：盘口新根。截面 t 只用 (t−30, t] 内的分钟（与平台根同一口径）。
  BK_SLOPEI 盘口斜率失衡：买侧 3 档量 ÷ (bp1 − bp3) 与卖侧 3 档量 ÷ (ap3 − ap1) 的 (买 − 卖) ÷ (买 + 卖)，段内均值
  BK_AOS    第 1 档平均单笔挂单量：(bv1 + av1) ÷ (买卖第 1 档委托笔数)，段内均值（再看 _DEV：相对 20 日均值）
  BK_RESIL  盘口恢复：中间价变动 ≥ 1 个价位的分钟之后，下一分钟第 1 档深度 ÷ 这一分钟第 1 档深度，段内均值
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
COLS = ["date", "instrument_id", "ask_price1", "ask_price3", "bid_price1", "bid_price3",
        "ask_volume1", "ask_volume2", "ask_volume3", "bid_volume1", "bid_volume2", "bid_volume3", "ask_num_orders1", "bid_num_orders1"]
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
    bv = d["bid_volume1"].to_numpy(float) + d["bid_volume2"].to_numpy(float) + d["bid_volume3"].to_numpy(float)
    av = d["ask_volume1"].to_numpy(float) + d["ask_volume2"].to_numpy(float) + d["ask_volume3"].to_numpy(float)
    ok = (bp1 > 0) & (ap1 > bp1) & (bp3 > 0) & (ap3 > 0) & (bp1 > bp3) & (ap3 > ap1)
    sb = np.where(ok, bv / np.where(ok, bp1 - bp3, 1.0), np.nan)
    sa = np.where(ok, av / np.where(ok, ap3 - ap1, 1.0), np.nan)
    slopei = (sb - sa) / (sb + sa)
    dep1 = d["bid_volume1"].to_numpy(float) + d["ask_volume1"].to_numpy(float)
    no1 = d["bid_num_orders1"].to_numpy(float) + d["ask_num_orders1"].to_numpy(float)
    aos = np.where(no1 > 0, dep1 / np.where(no1 > 0, no1, 1.0), np.nan)
    mid = np.where((bp1 > 0) & (ap1 > bp1), (ap1 + bp1) / 2.0, np.nan)
    key = [d["instrument_id"], d["day"]]
    mprev = pd.Series(mid).groupby(key, sort=False).shift(1).to_numpy()
    dnext = pd.Series(dep1).groupby(key, sort=False).shift(-1).to_numpy()
    jump = np.abs(mid - mprev) >= 1.0
    resil = np.where(jump & (dep1 > 0), dnext / np.where(dep1 > 0, dep1, 1.0), np.nan)
    # 截面 t 的段 (t−30, t]：resil 用到下一分钟，段内最后一分钟的下一分钟在段外 → 不用最后一分钟
    last = pd.Series(seg).ne(pd.Series(seg).shift(-1)).to_numpy()
    resil = np.where(last, np.nan, resil)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "BK_SLOPEI": slopei, "BK_AOS": aos, "BK_RESIL": resil})
    t = t[t["seg"] != ""]
    a = t.groupby(["iid", "day", "seg"], sort=False)[["BK_SLOPEI", "BK_AOS", "BK_RESIL"]].mean()
    return a.reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(12) as p:
        parts = p.map(one, files)
    A = pd.concat(parts, ignore_index=True)
    print("段行数", len(A), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    for nm in ("BK_SLOPEI", "BK_AOS", "BK_RESIL"):
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
