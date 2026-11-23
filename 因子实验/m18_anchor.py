# -*- coding: utf-8 -*-
"""纯方向深挖 J5（10-20）：日内锚定与盘口形态（1 分钟数据）。截面 t 只用当日到 t 为止的分钟。
  AN_VWAP   截面收盘相对当日至今 VWAP 的偏离 log(收盘 ÷ VWAP)（VWAP 是机构执行的锚，偏离后回归）
  AN_RPOS   截面收盘在当日至今价格区间中的位置 (收盘 − 最低) ÷ (最高 − 最低) − 0.5
  AN_MICRO  段内微观价格偏离的均值：(微观价格 − 中间价) ÷ 价差，微观价格 = (买一价·卖一量 + 卖一价·买一量) ÷ (买一量 + 卖一量)（Stoikov 2018）
  AN_SPRD   段内相对价差均值 ÷ 当日至今相对价差均值 − 1（流动性撤离）
  AN_CONC   段内成交量集中度：Σ(分钟量 ÷ 段量)²（脉冲式交易）
  *_20      日均值的过去 20 日均值（只用前几日）
保存 vc_tmp/h1_AN_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id", "high", "low", "close", "volume", "amount", "ask_price1", "bid_price1", "ask_volume1", "bid_volume1"]
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
NAMES = ["AN_VWAP", "AN_RPOS", "AN_MICRO", "AN_SPRD", "AN_CONC"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    keep = ((hm >= 931) & (hm <= 1130)) | ((hm >= 1301) & (hm <= 1500))
    d, seg = d[keep].reset_index(drop=True), seg[keep]
    v = lambda c: d[c].to_numpy(float)                                                       # noqa: E731
    gk = [d["instrument_id"], d["day"]]
    cum = lambda x: pd.Series(x).groupby(gk, sort=False).cumsum().to_numpy()                  # noqa: E731
    cmax = lambda x: pd.Series(x).groupby(gk, sort=False).cummax().to_numpy()                 # noqa: E731
    cmin = lambda x: pd.Series(x).groupby(gk, sort=False).cummin().to_numpy()                 # noqa: E731
    vol, amt, c = v("volume"), v("amount"), v("close")
    cv, ca = cum(vol), cum(amt)
    hi = cmax(np.where(v("high") > 0, v("high"), np.nan)); lo = cmin(np.where(v("low") > 0, v("low"), np.nan))
    ap, bp, av, bv = v("ask_price1"), v("bid_price1"), v("ask_volume1"), v("bid_volume1")
    okq = (ap > bp) & (bp > 0) & (av + bv > 0)
    mid = (ap + bp) / 2.0
    micro = np.where(okq, (bp * av + ap * bv) / np.where(okq, av + bv, 1.0), np.nan)
    rs = np.where(okq, (ap - bp) / np.where(okq, mid, 1.0), np.nan)
    rs_cum = cum(np.nan_to_num(rs)) / np.maximum(cum(np.isfinite(rs).astype(float)), 1.0)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "vw": np.where((cv > 0) & (c > 0), np.log(np.where(c > 0, c, 1.0) / np.where(cv > 0, ca / np.where(cv > 0, cv, 1.0), 1.0)), np.nan),
                      "rp": np.where(hi > lo, (c - lo) / np.where(hi > lo, hi - lo, 1.0) - 0.5, np.nan),
                      "mc": np.where(okq, (micro - mid) / np.where(okq, ap - bp, 1.0), np.nan),
                      "rs": rs, "rsc": rs_cum, "vol": vol})
    t = t[t["seg"] != ""]
    K = ["iid", "day", "seg"]
    t["vs2"] = (t["vol"] / t.groupby(K, sort=False)["vol"].transform("sum").replace(0, np.nan)) ** 2
    g = t.groupby(K, sort=False).agg(vw=("vw", "last"), rp=("rp", "last"), mc=("mc", "mean"), rs=("rs", "mean"), rsc=("rsc", "last"),
                                     conc=("vs2", "sum"), nv=("vs2", "count"))
    g["conc"] = g["conc"].where(g["nv"] > 0)
    out = pd.DataFrame(index=g.index)
    out["AN_VWAP"] = g["vw"]; out["AN_RPOS"] = g["rp"]; out["AN_MICRO"] = g["mc"]
    out["AN_SPRD"] = g["rs"] / g["rsc"].where(g["rsc"] > 0) - 1.0
    out["AN_CONC"] = g["conc"]
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(10) as p:
        parts = p.map(one, files)
    A = pd.concat(parts, ignore_index=True)
    print("段行数", len(A), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    for nm in NAMES:
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        slow = H1.roll_prev(Wd.mean(axis=1), 20, 10)
        S = pd.DataFrame({s0: slow.to_numpy() for s0 in H1.CS}, index=Wd.index)
        for tag, fr in ((nm, Wd), (nm + "_20", S)):
            x = H1.align(c, fr)
            x = np.where(np.isfinite(x) & (x == 0), 1e-9, x); x[z930] = 0.0
            np.save(H1.OUT / ("h1_%s.npy" % tag), np.nan_to_num(x, nan=0.0))
            print("保存", tag, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
