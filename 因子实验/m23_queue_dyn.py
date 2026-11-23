# -*- coding: utf-8 -*-
"""新信号源 N3（10-20 晚）：价格不变时一档挂单的增减（纯粹的排队与撤单行为）。相邻两分钟的一档快照：
  QB_NET    价格不变时买一挂单增量 − 卖一挂单增量，段内之和 ÷ 段内平均一档深度（只看挂单的进出，不含价格变动）
  QB_CNET   同上，用委托笔数（买一笔数增量 − 卖一笔数增量）÷ 平均一档笔数
  QB_CHURN  价格不变时两边挂单变化的绝对值之和 ÷ 平均深度（挂撤活跃度）
截面 t 只用 (t−30, t]。保存 vc_tmp/h1_QB_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
NAMES = ["QB_NET", "QB_CNET", "QB_CHURN"]
COLS = ["date", "instrument_id", "bid_price1", "ask_price1", "bid_volume1", "ask_volume1", "bid_num_orders1", "ask_num_orders1"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    gk = [d["instrument_id"], d["day"]]
    v = lambda c_: d[c_].to_numpy(float)                                                          # noqa: E731
    sh = lambda c_: pd.Series(v(c_)).groupby(gk, sort=False).shift(1).to_numpy()                   # noqa: E731
    bp, ap, bv, av, bn, an = (v(x) for x in COLS[2:])
    bp0, ap0, bv0, av0, bn0, an0 = (sh(x) for x in COLS[2:])
    sb = (bp == bp0) & (bp > 0); sa = (ap == ap0) & (ap > 0)
    dbv = np.where(sb, bv - bv0, 0.0); dav = np.where(sa, av - av0, 0.0)
    dbn = np.where(sb, bn - bn0, 0.0); dan = np.where(sa, an - an0, 0.0)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "net": dbv - dav, "cnet": dbn - dan, "churn": np.abs(dbv) + np.abs(dav),
                      "dep": np.where((bp > 0) & (ap > 0), (bv + av) / 2, np.nan), "cnt": np.where((bp > 0) & (ap > 0), (bn + an) / 2, np.nan)})
    t = t[t["seg"] != ""]
    g = t.groupby(["iid", "day", "seg"], sort=False).agg(net=("net", "sum"), cnet=("cnet", "sum"), churn=("churn", "sum"), dep=("dep", "mean"), cnt=("cnt", "mean"))
    out = pd.DataFrame(index=g.index)
    out["QB_NET"] = g["net"] / g["dep"].where(g["dep"] > 0)
    out["QB_CNET"] = g["cnet"] / g["cnt"].where(g["cnt"] > 0)
    out["QB_CHURN"] = g["churn"] / g["dep"].where(g["dep"] > 0)
    return out.replace([np.inf, -np.inf], np.nan).reset_index()


def main():
    with Pool(10) as p:
        parts = p.map(one, sorted(glob.glob(SRC)))
    A = pd.concat(parts, ignore_index=True)
    print("段行数", len(A), flush=True)
    c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
    for nm in NAMES:
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        x = H1.align(c, Wd)
        x = np.where(np.isfinite(x) & (x == 0), 1e-9, x); x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
