# -*- coding: utf-8 -*-
"""方向 A（10-20）：分解式反转的原料。把每个半小时段 (t−30, t] 的 1 分钟对数收益拆成几块，每块含义不同：
  D_R      段收益（全部分钟之和）
  D_JUMP   跳跃分钟的收益：|r| > 3·√(BV/(n−1))，BV = π/2·Σ|r_t||r_t−1|（Barndorff-Nielsen–Shephard；跳跃 = 信息冲击）
  D_HV     放量分钟的收益：段内成交量排前三分之一的分钟（Campbell–Grossman–Wang 1993：伴随放量的价格变动更多是流动性冲击）
  D_LV     其余分钟的收益（缩量时的价格漂移）
  C_*      同上，当日开盘到截面的累计（各段之和）
假设：流动性冲击、噪声那部分反转，信息那部分延续或反转弱。保存 vc_tmp/h1_D_*.npy、h1_C_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id", "close", "volume"]
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
PARTS = ["D_R", "D_JUMP", "D_HV", "D_LV"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    c = d["close"].to_numpy(float)
    pc = pd.Series(c).groupby([d["instrument_id"], d["day"]], sort=False).shift(1).to_numpy()
    r = np.where((c > 0) & (pc > 0), np.log(np.where(c > 0, c, 1.0) / np.where(pc > 0, pc, 1.0)), np.nan)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "r": r, "v": d["volume"].to_numpy(float)})
    t = t[(t["seg"] != "") & np.isfinite(t["r"])].reset_index(drop=True)
    K = ["iid", "day", "seg"]
    g = t.groupby(K, sort=False)
    t["bp"] = np.abs(t["r"]) * np.abs(g["r"].shift(1))
    n = g["r"].transform("size")
    thr = 3.0 * np.sqrt(np.pi / 2 * g["bp"].transform("sum") / (n - 1).clip(lower=1))
    jump = (np.abs(t["r"]) > thr) & (n >= 10)
    hv = g["v"].rank(method="first", ascending=False) <= (n / 3.0)
    t["rj"] = np.where(jump, t["r"], 0.0)
    t["rh"] = np.where(hv & ~jump, t["r"], 0.0)
    t["rl"] = np.where(~hv & ~jump, t["r"], 0.0)
    a = t.groupby(K, sort=False).agg(D_R=("r", "sum"), D_JUMP=("rj", "sum"), D_HV=("rh", "sum"), D_LV=("rl", "sum"))
    return a.reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(12) as p:
        parts = p.map(one, files)
    A = pd.concat(parts, ignore_index=True)
    print("段行数", len(A), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    for nm in PARTS:
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        for tag, fr in ((nm, Wd), ("C_" + nm[2:], Wd.fillna(0.0).cumsum(axis=1).where(Wd.notna()))):
            x = H1.align(c, fr)
            x[z930] = 0.0
            np.save(H1.OUT / ("h1_%s.npy" % tag), np.nan_to_num(x, nan=0.0))
            print("保存", tag, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
