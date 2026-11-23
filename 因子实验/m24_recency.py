# -*- coding: utf-8 -*-
"""新信号源 N5（10-20 晚）：冲击发生在段内的什么时候。把段 (t−30, t] 的 1 分钟对数收益拆成：
  RC_LAST5   最后 5 分钟的收益（越接近截面的冲击）
  RC_FIRST25 前 25 分钟的收益
  RC_LAST10  最后 10 分钟的收益
  RC_PATH    段内价格路径的"末端偏离"：收盘相对段内均价的对数偏离（冲击后有没有已经部分回吐）
保存 vc_tmp/h1_RC_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
NAMES = ["RC_LAST5", "RC_FIRST25", "RC_LAST10", "RC_PATH"]


def one(f):
    d = pd.read_feather(f, columns=["date", "instrument_id", "close"])
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object); last5 = np.zeros(len(d), bool); last10 = np.zeros(len(d), bool)
    for a, b, s in EDGES:
        m = (hm >= a) & (hm <= b)
        seg[m] = s; last5 |= m & (hm > b - 5); last10 |= m & (hm > b - 10)
    c = d["close"].to_numpy(float)
    pc = pd.Series(c).groupby([d["instrument_id"], d["day"]], sort=False).shift(1).to_numpy()
    r = np.where((c > 0) & (pc > 0), np.log(np.where(c > 0, c, 1.0) / np.where(pc > 0, pc, 1.0)), np.nan)
    lc = np.log(np.where(c > 0, c, np.nan))
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "r5": np.where(last5, r, 0.0), "r25": np.where(~last5, r, 0.0), "r10": np.where(last10, r, 0.0), "lc": lc, "ok": np.isfinite(r)})
    t = t[(t["seg"] != "") & t["ok"]]
    g = t.groupby(["iid", "day", "seg"], sort=False).agg(RC_LAST5=("r5", "sum"), RC_FIRST25=("r25", "sum"), RC_LAST10=("r10", "sum"), lcl=("lc", "last"), lcm=("lc", "mean"))
    g["RC_PATH"] = g["lcl"] - g["lcm"]
    return g[NAMES].reset_index()


def main():
    with Pool(10) as p:
        parts = p.map(one, sorted(glob.glob(SRC)))
    A = pd.concat(parts, ignore_index=True)
    c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
    for nm in NAMES:
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        x = H1.align(c, Wd)
        x = np.where(np.isfinite(x) & (x == 0), 1e-9, x); x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
