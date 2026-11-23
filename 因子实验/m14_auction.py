# -*- coding: utf-8 -*-
"""新信息源（10-20）：开盘集合竞价（09:16～09:25）。现有分量都从 09:30 之后开始，没有用过竞价阶段的盘口。
1 分钟表在 09:16～09:24 每分钟给出：收盘价 = 虚拟成交价，买一／卖一量 = 虚拟匹配量，买二或卖二量 = 未匹配的买单或卖单；09:25 = 竞价结果。
日内不变的信号（09:25 已知，当日各截面相同）：
  AU_IMB    09:24 未匹配（买 − 卖）÷ 匹配量（开盘买卖压力）
  AU_LATE   09:20 → 09:25 虚拟价变化（09:20 后不能撤单，反映真实意图）
  AU_EARLY  09:16 → 09:20 虚拟价变化（可撤单阶段，可能有试探或虚假挂单）
  AU_MGROW  09:20 → 09:24 匹配量的对数增长（参与的坚定程度）
  AU_VOLR   今日竞价成交额 ÷ 近 20 日竞价成交额均值（关注度，只用前几日作分母）
  AU_SIZE   竞价平均单笔额 ÷ 昨日 09:31 之后的平均单笔额（机构参与竞价的程度）
保存 vc_tmp/h1_AU_*.npy（原始值）。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id", "close", "volume", "amount", "deal_number", "ask_volume1", "ask_volume2", "bid_volume2"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    hm = d["date"].dt.hour * 100 + d["date"].dt.minute
    d["day"] = d["date"].dt.normalize(); d["hm"] = hm
    k = ["instrument_id", "day"]
    pick = lambda h, cols: d[d["hm"] == h].set_index(k)[cols]           # noqa: E731
    a16, a20, a24, a25 = pick(916, ["close"]), pick(920, ["close", "ask_volume1"]), pick(924, ["close", "ask_volume1", "ask_volume2", "bid_volume2"]), pick(925, ["close", "amount", "deal_number"])
    cont = d[d["hm"] >= 931].groupby(k)[["amount", "deal_number"]].sum()
    o = pd.DataFrame(index=a25.index)
    o["p16"] = a16["close"]; o["p20"] = a20["close"]; o["m20"] = a20["ask_volume1"]
    o["p24"] = a24["close"]; o["m24"] = a24["ask_volume1"]; o["ub"] = a24["bid_volume2"]; o["ua"] = a24["ask_volume2"]
    o["p25"] = a25["close"]; o["amt25"] = a25["amount"]; o["dn25"] = a25["deal_number"]
    o["camt"] = cont["amount"]; o["cdn"] = cont["deal_number"]
    return o.reset_index()


def main():
    files = sorted(glob.glob(SRC))
    with Pool(12) as p:
        parts = p.map(one, files)
    A = pd.concat(parts, ignore_index=True).rename(columns={"instrument_id": "iid"}).set_index(["iid", "day"]).sort_index()
    A = A[~A.index.duplicated(keep="last")]
    g = lambda s: s.groupby(level=0, sort=False)                          # noqa: E731
    pos = lambda s: s.where(s > 0)                                        # noqa: E731
    S = {}
    S["AU_IMB"] = (A["ub"] - A["ua"]) / pos(A["m24"])
    S["AU_LATE"] = A["p25"] / pos(A["p20"]) - 1.0
    S["AU_EARLY"] = A["p20"] / pos(A["p16"]) - 1.0
    S["AU_MGROW"] = np.log(pos(A["m24"]) / pos(A["m20"]))
    S["AU_VOLR"] = A["amt25"] / pos(H1.roll_prev(A["amt25"], 20, 10))
    S["AU_SIZE"] = (A["amt25"] / pos(A["dn25"])) / pos(g(A["camt"] / pos(A["cdn"])).shift(1))
    c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
    for nm, ser in S.items():
        ser = ser.replace([np.inf, -np.inf], np.nan)
        fr = pd.DataFrame({s0: ser.to_numpy() for s0 in H1.CS}, index=ser.index)
        v = H1.align(c, fr)
        v = np.where(np.isfinite(v) & (v == 0), 1e-9, v)                   # 真实的 0 保留为极小值，避免被当作缺失
        v[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(v) != 0).mean()), "中位数 %.4f" % float(np.nanmedian(ser)), flush=True)


if __name__ == "__main__":
    main()
