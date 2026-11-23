# -*- coding: utf-8 -*-
"""可解释信号库第三批（10-20）：1 分钟微观结构。截面 t 只用 (t−30, t] 内的分钟（与平台根同一口径）；*_20 = 过去 20 日日均值的均值（只用前几日）。
  MS_VR     方差比：段内 5 分钟收益平方和 ÷ 1 分钟收益平方和（Lo–MacKinlay 1988；<1 = 分钟级回复，流动性提供；>1 = 趋势，知情交易）
  MS_AC1    1 分钟收益一阶自相关（Roll 1984：负自相关 ≈ 买卖价差弹跳）
  MS_JUMP   跳跃占比：max(0, RV − BV) ÷ RV，BV = π/2·Σ|r_t||r_t−1|（Barndorff-Nielsen–Shephard 2004；跳跃 = 信息冲击）
  MS_AMI    段内 Amihud：Σ|r| ÷ Σ成交额（价格冲击，Kyle 1985 λ 的代理）
  MS_SMART  聪明钱 Q：按 |r|/√量 从大到小取到 20% 成交量的分钟，其 VWAP ÷ 段内 VWAP（大单冲击价位相对均价）
  MS_OFI    逐笔方向失衡：Σsign(r)·量 ÷ Σ量（tick rule 方向，主动买压）
保存 vc_tmp/h1_MS_*.npy（原始值）。数据：macstudio 的原始 1 分钟 feather，12 进程并行。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
COLS = ["date", "instrument_id", "close", "volume", "amount"]
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430")]
NAMES = ["MS_VR", "MS_AC1", "MS_JUMP", "MS_AMI", "MS_SMART", "MS_OFI"]


def one(f):
    d = pd.read_feather(f, columns=COLS)
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    h, m = d["date"].dt.hour.to_numpy(), d["date"].dt.minute.to_numpy()
    hm = h * 100 + m
    seg = np.full(len(d), "", dtype=object)
    for a, b, s in EDGES:
        seg[(hm >= a) & (hm <= b)] = s
    o = np.where(h < 12, h * 60 + m - 570, h * 60 + m - 780 + 120)      # 09:31 → 1，13:01 → 121
    c = d["close"].to_numpy(float)
    gk = [d["instrument_id"], d["day"]]
    pc = pd.Series(c).groupby(gk, sort=False).shift(1).to_numpy()
    r = np.where((c > 0) & (pc > 0), np.log(np.where(c > 0, c, 1.0) / np.where(pc > 0, pc, 1.0)), np.nan)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "blk": (o - 1) // 5, "r": r, "v": d["volume"].to_numpy(float), "a": d["amount"].to_numpy(float)})
    t = t[(t["seg"] != "") & np.isfinite(t["r"])].reset_index(drop=True)
    K = ["iid", "day", "seg"]
    t["rl"] = t.groupby(K, sort=False)["r"].shift(1)
    t["r2"] = t["r"] ** 2
    t["rr"] = t["r"] * t["rl"]
    t["bp"] = np.abs(t["r"]) * np.abs(t["rl"])
    t["ar"] = np.abs(t["r"])
    t["sv"] = np.sign(t["r"]) * t["v"]
    r5 = t.groupby(K + ["blk"], sort=False)["r"].sum().pow(2).groupby(level=[0, 1, 2], sort=False).sum().rename("rv5")
    t["S"] = t["ar"] / np.sqrt(t["v"].where(t["v"] > 0))                  # 聪明钱：|r|/√量 从大到小，累计量占段内 20% 以内
    t = t.sort_values(K + ["S"], ascending=[True, True, True, False], na_position="last")
    tv = t.groupby(K, sort=False)["v"].transform("sum")
    cs = t.groupby(K, sort=False)["v"].cumsum() - t["v"]
    sm = (cs < 0.2 * tv) & t["S"].notna()
    t["vs"], t["as_"] = np.where(sm, t["v"], 0.0), np.where(sm, t["a"], 0.0)
    g = t.groupby(K, sort=False).agg(r2=("r2", "sum"), rr=("rr", "sum"), bp=("bp", "sum"), ar=("ar", "sum"), a=("a", "sum"),
                                     v=("v", "sum"), sv=("sv", "sum"), vs=("vs", "sum"), as_=("as_", "sum"), n=("r", "size"))
    g = g.join(r5)
    ok = (g["r2"] > 0) & (g["n"] >= 10)
    out = pd.DataFrame(index=g.index)
    out["MS_VR"] = (g["rv5"] / g["r2"]).where(ok)
    out["MS_AC1"] = (g["rr"] / g["r2"]).where(ok)
    out["MS_JUMP"] = ((g["r2"] - np.pi / 2 * g["bp"]).clip(lower=0) / g["r2"]).where(ok)
    out["MS_AMI"] = np.log(g["ar"] / g["a"].where(g["a"] > 0) * 1e8 + 1e-6)
    out["MS_SMART"] = ((g["as_"] / g["vs"].where(g["vs"] > 0)) / (g["a"] / g["v"].where(g["v"] > 0))).where(g["v"] > 0)
    out["MS_OFI"] = (g["sv"] / g["v"].where(g["v"] > 0))
    return out.reset_index()


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
        Wd = Wd.replace([np.inf, -np.inf], np.nan)
        slow = H1.roll_prev(Wd.mean(axis=1), 20, 10)
        S = pd.DataFrame({s0: slow.to_numpy() for s0 in H1.CS}, index=Wd.index)
        for tag, fr in ((nm, Wd), (nm + "_20", S)):
            x = H1.align(c, fr)
            x[z930] = 0.0
            np.save(H1.OUT / ("h1_%s.npy" % tag), np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
            print("保存", tag, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
