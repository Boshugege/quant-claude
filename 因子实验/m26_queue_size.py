# -*- coding: utf-8 -*-
"""T11 的特征（10-22）：一档平均委托规模比 qs1 = ln[(买一量/买一笔数)/(卖一量/卖一笔数)]（逐分钟），
  QS1L  截面前 10 分钟的均值（= 评分机 s20_ql_s1only 的原值，未取负、未平滑）
  QS1A  整段（30 分钟）均值
截面 t 只用 (t−30, t]。保存 vc_tmp/h1_QS1L.npy、h1_QS1A.npy（原始值）。写法同 m23。"""
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
NAMES = ["QS1L", "QS1A"]
COLS = ["date", "instrument_id", "bid_volume1", "ask_volume1", "bid_num_orders1", "ask_num_orders1"]


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
    bv, av, bn, an = (d[c_].to_numpy(float) for c_ in COLS[2:])
    ok = (bv > 0) & (av > 0) & (bn > 0) & (an > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(ok, np.log((bv / bn) / (av / an)), np.nan)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "qa": q, "ql": np.where(last, q, np.nan)})
    t = t[t["seg"] != ""]
    g = t.groupby(["iid", "day", "seg"], sort=False).agg(QS1L=("ql", "mean"), QS1A=("qa", "mean"))
    return g.replace([np.inf, -np.inf], np.nan).reset_index()


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
