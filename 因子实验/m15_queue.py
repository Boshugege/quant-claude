# -*- coding: utf-8 -*-
"""新信息源（10-20）：盘口委托笔数（ask／bid_num_orders 1～3）。现有分量只用过盘口的价与量，没用过笔数。
截面 t 只用 (t−30, t] 内的分钟（段内均值）；*_20 = 过去 20 日日均值的均值（只用前几日）。
  OQ_NIMB   一档委托笔数失衡：(买一笔数 − 卖一笔数) ÷ 二者之和（散户排队的方向）
  OQ_SIZE   一档平均单笔挂单规模之比：log[(买一量 ÷ 买一笔数) ÷ (卖一量 ÷ 卖一笔数)]（大单托底 vs 大单压盘）
  OQ_DSIZE  1～3 档合计的平均单笔挂单规模之比（同上，更深）
  OQ_AVG    双边一档平均单笔挂单规模 ÷ 该股过去 20 日均值（盘口上大单是否异常多）
保存 vc_tmp/h1_OQ_*.npy（原始值）。数据：macstudio 的原始 1 分钟 feather，12 进程并行。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id"] + ["%s_%s%d" % (s, k, i) for s in ("ask", "bid") for k in ("volume", "num_orders") for i in (1, 2, 3)]
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
FAST = ["OQ_NIMB", "OQ_SIZE", "OQ_DSIZE"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    v = lambda c: d[c].to_numpy(float)                                   # noqa: E731
    bn1, an1, bv1, av1 = v("bid_num_orders1"), v("ask_num_orders1"), v("bid_volume1"), v("ask_volume1")
    bn, an = bn1 + v("bid_num_orders2") + v("bid_num_orders3"), an1 + v("ask_num_orders2") + v("ask_num_orders3")
    bv, av = bv1 + v("bid_volume2") + v("bid_volume3"), av1 + v("ask_volume2") + v("ask_volume3")
    ok1 = (bn1 > 0) & (an1 > 0) & (bv1 > 0) & (av1 > 0)
    okd = (bn > 0) & (an > 0) & (bv > 0) & (av > 0)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["date"].dt.normalize().to_numpy(), "seg": seg,
                      "OQ_NIMB": np.where(ok1, (bn1 - an1) / np.where(ok1, bn1 + an1, 1.0), np.nan),
                      "OQ_SIZE": np.where(ok1, np.log(np.where(ok1, (bv1 / np.where(ok1, bn1, 1.0)) / (av1 / np.where(ok1, an1, 1.0)), 1.0)), np.nan),
                      "OQ_DSIZE": np.where(okd, np.log(np.where(okd, (bv / np.where(okd, bn, 1.0)) / (av / np.where(okd, an, 1.0)), 1.0)), np.nan),
                      "avg1": np.where(ok1, (bv1 + av1) / np.where(ok1, bn1 + an1, 1.0), np.nan)})
    t = t[t["seg"] != ""]
    return t.groupby(["iid", "day", "seg"], sort=False)[FAST + ["avg1"]].mean().reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(12) as p:
        parts = p.map(one, files)
    A = pd.concat(parts, ignore_index=True)
    print("段行数", len(A), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    def save(tag, fr):
        x = H1.align(c, fr)
        x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % tag), np.nan_to_num(x, nan=0.0))
        print("保存", tag, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)
    for nm in FAST + ["avg1"]:
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        slow = H1.roll_prev(Wd.mean(axis=1), 20, 10)
        S = pd.DataFrame({s0: slow.to_numpy() for s0 in H1.CS}, index=Wd.index)
        if nm == "avg1":
            save("OQ_AVG", np.log(Wd / S.to_numpy()))
        else:
            save(nm, Wd)
            save(nm + "_20", S)


if __name__ == "__main__":
    main()
