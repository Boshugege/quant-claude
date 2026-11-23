# -*- coding: utf-8 -*-
"""新信号源 N1（10-20 晚）：按成交单笔大小拆分的段收益。每分钟平均成交单笔额 = 成交额 ÷ 成交笔数（大 = 大户／知情，小 = 散户／噪音），
与该股当日至今分钟单笔额的中位数比较，把段 (t−30, t] 的 1 分钟对数收益拆成：
  TS_BIG    单笔额高于当日中位数 × 1.5 的分钟的收益（大单推动）
  TS_SMALL  单笔额低于当日中位数 × 0.67 的分钟的收益（小单推动）
  TS_MID    其余分钟
  TS_BIGSH  大单分钟的成交额占段内成交额的比例（大户参与度）
文献：Easley–O'Hara 1987、Barber–Odean 2008（散户噪音交易回吐，知情大单延续）。保存 vc_tmp/h1_TS_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
NAMES = ["TS_BIG", "TS_SMALL", "TS_MID", "TS_BIGSH"]


def one(f):
    d = pd.read_feather(f, columns=["date", "instrument_id", "close", "amount", "deal_number"])
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    gk = [d["instrument_id"], d["day"]]
    c = d["close"].to_numpy(float)
    pc = pd.Series(c).groupby(gk, sort=False).shift(1).to_numpy()
    r = np.where((c > 0) & (pc > 0), np.log(np.where(c > 0, c, 1.0) / np.where(pc > 0, pc, 1.0)), np.nan)
    amt, dn = d["amount"].to_numpy(float), d["deal_number"].to_numpy(float)
    ats = np.where(dn > 0, amt / np.where(dn > 0, dn, 1.0), np.nan)               # 分钟平均成交单笔额
    med = pd.Series(ats).groupby(gk, sort=False).expanding().median().reset_index(level=[0, 1], drop=True).sort_index().to_numpy()   # 当日至今中位数
    big = ats > 1.5 * med; small = ats < 0.67 * med
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg, "r": r, "amt": amt,
                      "rb": np.where(big, r, 0.0), "rs": np.where(small, r, 0.0), "rm": np.where(~big & ~small, r, 0.0),
                      "ab": np.where(big, amt, 0.0)})
    t = t[(t["seg"] != "") & np.isfinite(t["r"])]
    g = t.groupby(["iid", "day", "seg"], sort=False).agg(TS_BIG=("rb", "sum"), TS_SMALL=("rs", "sum"), TS_MID=("rm", "sum"), ab=("ab", "sum"), amt=("amt", "sum"))
    g["TS_BIGSH"] = g["ab"] / g["amt"].where(g["amt"] > 0)
    return g[NAMES].reset_index()


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
