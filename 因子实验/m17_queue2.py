# -*- coding: utf-8 -*-
"""纯方向深挖 J2（10-20）：委托结构（m15 的延伸）。截面 t 只用 (t−30, t] 内的分钟；*_20 = 过去 20 日日均值的均值。
  QL_NIMB23  第 2、3 档合计的委托笔数失衡 (买 − 卖) ÷ (买 + 卖)（排在后面的队伍）
  QL_NIMBA   1～3 档合计的笔数失衡
  QL_ARR     新增委托净流入：段内各分钟 Δ(买 1～3 档笔数) − Δ(卖 1～3 档笔数) 之和 ÷ 段内平均总笔数（排队人数的增减方向）
  QL_RTS     挂单单笔规模 ÷ 成交单笔规模：一档 (买量 + 卖量) ÷ (买笔数 + 卖笔数)，除以 段成交量 ÷ 段成交笔数（挂的单比成交的单大 = 做市／大户挂单）
  QL_LATE    段内最后 10 分钟的一档笔数失衡（最近的排队方向）
  QL_SLATE   段内最后 10 分钟的一档单笔规模比 log(买 ÷ 卖)
保存 vc_tmp/h1_QL_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id", "volume", "deal_number"] + ["%s_%s%d" % (s, k, i) for s in ("ask", "bid") for k in ("volume", "num_orders") for i in (1, 2, 3)]
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
NAMES = ["QL_NIMB23", "QL_NIMBA", "QL_ARR", "QL_RTS", "QL_LATE", "QL_SLATE"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    late = np.zeros(len(d), bool)
    for a, b, s in EDGES:
        m = (hm >= a) & (hm <= b)
        seg[m] = s
        late |= m & (hm > b - 10)
    v = lambda c: d[c].to_numpy(float)                                                       # noqa: E731
    bn1, an1, bv1, av1 = v("bid_num_orders1"), v("ask_num_orders1"), v("bid_volume1"), v("ask_volume1")
    bn23, an23 = v("bid_num_orders2") + v("bid_num_orders3"), v("ask_num_orders2") + v("ask_num_orders3")
    bnA, anA = bn1 + bn23, an1 + an23
    gk = [d["instrument_id"], d["day"]]
    dB = pd.Series(bnA).groupby(gk, sort=False).diff().to_numpy(); dA = pd.Series(anA).groupby(gk, sort=False).diff().to_numpy()
    imb = lambda b, a: np.where((b + a) > 0, (b - a) / np.where((b + a) > 0, b + a, 1.0), np.nan)   # noqa: E731
    ok1 = (bn1 > 0) & (an1 > 0) & (bv1 > 0) & (av1 > 0)
    sz = np.where(ok1, np.log(np.where(ok1, (bv1 / np.where(ok1, bn1, 1.0)) / (av1 / np.where(ok1, an1, 1.0)), 1.0)), np.nan)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "n23": imb(bn23, an23), "nA": imb(bnA, anA), "arr": dB - dA, "tot": bnA + anA,
                      "rest": np.where(ok1, (bv1 + av1) / np.where(ok1, bn1 + an1, 1.0), np.nan),
                      "vol": v("volume"), "dn": v("deal_number"),
                      "n1L": np.where(late, imb(bn1, an1), np.nan), "szL": np.where(late, sz, np.nan)})
    t = t[t["seg"] != ""]
    g = t.groupby(["iid", "day", "seg"], sort=False).agg(n23=("n23", "mean"), nA=("nA", "mean"), arr=("arr", "sum"), tot=("tot", "mean"),
                                                         rest=("rest", "mean"), vol=("vol", "sum"), dn=("dn", "sum"), n1L=("n1L", "mean"), szL=("szL", "mean"))
    out = pd.DataFrame(index=g.index)
    out["QL_NIMB23"] = g["n23"]; out["QL_NIMBA"] = g["nA"]
    out["QL_ARR"] = g["arr"] / g["tot"].where(g["tot"] > 0)
    out["QL_RTS"] = np.log(g["rest"] / (g["vol"] / g["dn"].where(g["dn"] > 0)))
    out["QL_LATE"] = g["n1L"]; out["QL_SLATE"] = g["szL"]
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(12) as p:
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
