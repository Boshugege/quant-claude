# -*- coding: utf-8 -*-
"""纯方向深挖 J1（10-20）：订单流失衡（Cont–Kukanov–Stoikov 2014 的 OFI；多档版本 Xu–Gould–Howison 2019 MLOFI）。
用相邻两分钟的盘口快照（分钟末）计算每分钟的 OFI：
  e = 1[bp ≥ bp'] · bv − 1[bp ≤ bp'] · bv' − 1[ap ≤ ap'] · av + 1[ap ≥ ap'] · av'（' = 上一分钟），涵盖挂单、撤单与成交对盘口的净冲击。
截面 t 只用 (t−30, t] 内的分钟：
  CF_OFI1   一档 OFI 段内之和 ÷ 段内平均一档深度（买一量 + 卖一量）/ 2
  CF_OFI3   1～3 档 OFI 之和 ÷ 段内平均 1～3 档深度
  CF_DIV    一档 OFI 的截面秩 − 段收益的截面秩（买压大而价格没涨 = 被动卖方吸收；反之 = 价格被少量订单推动）
  CF_TICK   逐笔方向买压（= MS_OFI，对照）
  *_20      日均值的过去 20 日均值（只用前几日）
保存 vc_tmp/h1_CF_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R
import d2_optimize as O

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id", "close", "volume"] + ["%s_%s%d" % (s, k, i) for s in ("ask", "bid") for k in ("price", "volume") for i in (1, 2, 3)]
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
NAMES = ["CF_OFI1", "CF_OFI3", "CF_RET", "CF_TICK"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    gk = [d["instrument_id"], d["day"]]
    sh = lambda c: pd.Series(d[c].to_numpy(float)).groupby(gk, sort=False).shift(1).to_numpy()   # noqa: E731
    v = lambda c: d[c].to_numpy(float)                                                         # noqa: E731
    ofi, dep = {}, {}
    for i in (1, 2, 3):
        bp, bv, ap, av = v("bid_price%d" % i), v("bid_volume%d" % i), v("ask_price%d" % i), v("ask_volume%d" % i)
        bp0, bv0, ap0, av0 = sh("bid_price%d" % i), sh("bid_volume%d" % i), sh("ask_price%d" % i), sh("ask_volume%d" % i)
        ok = (bp > 0) & (ap > 0) & (bp0 > 0) & (ap0 > 0)
        e = (bp >= bp0) * bv - (bp <= bp0) * bv0 - (ap <= ap0) * av + (ap >= ap0) * av0
        ofi[i] = np.where(ok, e, np.nan)
        dep[i] = np.where((bp > 0) & (ap > 0), (bv + av) / 2.0, np.nan)
    c = v("close"); pc = sh("close")
    r = np.where((c > 0) & (pc > 0), np.log(np.where(c > 0, c, 1.0) / np.where(pc > 0, pc, 1.0)), np.nan)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "o1": ofi[1], "o3": ofi[1] + ofi[2] + ofi[3], "d1": dep[1], "d3": dep[1] + dep[2] + dep[3],
                      "r": r, "sv": np.sign(np.nan_to_num(r)) * v("volume"), "v": v("volume")})
    t = t[t["seg"] != ""]
    g = t.groupby(["iid", "day", "seg"], sort=False).agg(o1=("o1", "sum"), o3=("o3", "sum"), d1=("d1", "mean"), d3=("d3", "mean"),
                                                         r=("r", "sum"), sv=("sv", "sum"), v=("v", "sum"), n=("o1", "count"))
    out = pd.DataFrame(index=g.index)
    out["CF_OFI1"] = (g["o1"] / g["d1"].where(g["d1"] > 0)).where(g["n"] >= 10)
    out["CF_OFI3"] = (g["o3"] / g["d3"].where(g["d3"] > 0)).where(g["n"] >= 10)
    out["CF_RET"] = g["r"]
    out["CF_TICK"] = g["sv"] / g["v"].where(g["v"] > 0)
    return out.reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(12) as p:
        parts = p.map(one, files)
    A = pd.concat(parts, ignore_index=True)
    print("段行数", len(A), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    al = {}
    for nm in NAMES:
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        Wd = Wd.replace([np.inf, -np.inf], np.nan)
        al[nm] = H1.align(c, Wd)
        if nm != "CF_RET":
            slow = H1.roll_prev(Wd.mean(axis=1), 20, 10)
            S = pd.DataFrame({s0: slow.to_numpy() for s0 in H1.CS}, index=Wd.index)
            al[nm + "_20"] = H1.align(c, S)
    rk = lambda x: O.rc_of(x, c.date_id, c.n_dates)                                          # noqa: E731
    al["CF_DIV"] = rk(al["CF_OFI1"]) - rk(al["CF_RET"])
    for nm, x in al.items():
        if nm == "CF_RET":
            continue
        x = np.where(np.isfinite(x) & (x == 0), 1e-9, x); x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
