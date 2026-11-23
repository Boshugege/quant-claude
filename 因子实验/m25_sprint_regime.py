# -*- coding: utf-8 -*-
"""给 1617 的树加新特征（10-21，用户："1617 的逻辑你能进一步思考和批判吗？冲刺确实不错，再看看有没有更好的优化方式"）。
1617 是 27 个现成信号的排名堆叠：看不到超短线信息（冲刺相对 1617 残差 IC 0.027），也看不到市况（压力被高离散半卡住）。
本脚本从 1 分钟数据（按月文件，含复权因子；跨日用复权价，昨收 = 前一交易日 15:00 段的复权收盘）算两类特征，保存 vc_tmp/h1_*.npy：
  冲刺类（逐截面排名后进树）：
    SP_LSD   −(本段收益 − 当天至今收益 / 8)                     （评分机 s10_last_seg_vs_day 的口径）
    SP_LSDP  −(本段收益 − 当天至今收益 / 已过段数)               （修正：10:00 只过了 1 段）
    SP_LSDV  −(本段 VWAP 收益 − 当天至今 VWAP 收益 / 已过段数)    （用段 VWAP，降噪）
    SP_ACC   −(本段收益 − 上一段收益)                            （同一交易日内；10:00 记缺失）
    SP_CLV   −段内收盘位置 (2·收 − 高 − 低)/(高 − 低)
    SP_A1000 −ln(现价 / 10:00 价)（10:30 起；时间锚点）
  市况类（同一截面相同，**不排名**：h2_robust 对 MKT_／MKD_ 开头的特征用原值）：
    MKT_DDISP 当天至今收益的截面标准差（本段收益的截面均值／标准差已有 m1_market 的 MKT_SEG／MKT_DISP）
只用截面时刻及之前的数据。"""
import glob
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar1m/cpt_jyc_2026_e2e_bar1m/*.feather"
EDGES = [(931, 1000, "1000"), (1001, 1030, "1030"), (1031, 1100, "1100"), (1101, 1130, "1130"),
         (1301, 1330, "1330"), (1331, 1400, "1400"), (1401, 1430, "1430"), (1431, 1500, "1500")]   # 1500 段只用来取当日收盘（昨收）
NSEG = {"1000": 1, "1030": 2, "1100": 3, "1130": 4, "1330": 5, "1400": 6, "1430": 7, "1500": 8}
RANKED = ["SP_LSD", "SP_LSDP", "SP_LSDV", "SP_ACC", "SP_CLV", "SP_A1000", "SP_RSEG", "SP_RDAY"]
RAW = ["MKT_DDISP"]


def one(f):
    """每个月份文件 → 段级表：复权后的段末收盘、段内高低（原价，只用于段内位置）、段内清洗后的成交额与量。"""
    d = pd.read_feather(f, columns=["date", "instrument_id", "adjust_factor", "high", "low", "close", "volume", "amount"])
    d["day"] = d["date"].dt.normalize()
    d = d.sort_values(["instrument_id", "date"]).reset_index(drop=True)
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    seg = np.full(len(d), "", dtype=object)
    for a, b, s_ in EDGES:
        seg[(hm >= a) & (hm <= b)] = s_
    c = d["close"].to_numpy(float); adj = d["adjust_factor"].to_numpy(float)
    ok = (c > 0) & np.isfinite(adj) & (adj > 0)
    v, a_ = d["volume"].to_numpy(float), d["amount"].to_numpy(float)
    lo, hi = d["low"].to_numpy(float), d["high"].to_numpy(float)
    okv = (v > 0) & (a_ > 0) & (lo > 0) & (a_ / np.where(v > 0, v, 1) >= lo * 0.98) & (a_ / np.where(v > 0, v, 1) <= hi * 1.02)
    t = pd.DataFrame({"iid": d["instrument_id"].to_numpy(np.int64), "day": d["day"].to_numpy(), "seg": seg,
                      "c": np.where(ok, c, np.nan), "ca": np.where(ok, c * adj, np.nan), "adj": np.where(ok, adj, np.nan),
                      "h": hi, "l": lo, "va": np.where(okv, a_, 0.0), "vv": np.where(okv, v, 0.0)})
    t = t[t["seg"] != ""]
    return t.groupby(["iid", "day", "seg"], sort=False).agg(c=("c", "last"), ca=("ca", "last"), adj=("adj", "last"), h=("h", "max"), l=("l", "min"),
                                                          va=("va", "sum"), vv=("vv", "sum")).reset_index()


def features(g):
    g = g.sort_values(["iid", "day", "seg"]).reset_index(drop=True)
    dc = g[g["seg"] == "1500"].set_index(["iid", "day"])["ca"]                 # 当日复权收盘
    dc = dc.groupby(level=0).shift(1).rename("pca")                            # 前一交易日复权收盘 = 昨收
    g = g.merge(dc.reset_index(), on=["iid", "day"], how="left")
    gd = g.groupby(["iid", "day"], sort=False)
    pca = g["pca"].where(g["pca"] > 0)
    prev = gd["ca"].shift(1).fillna(pca)                                      # 10:00 段的上一价 = 昨收
    r_seg = np.log(g["ca"] / prev)
    r_day = np.log(g["ca"] / pca)
    nseg = g["seg"].map(NSEG).astype(float)
    svw = g["va"] / g["vv"].where(g["vv"] > 0) * g["adj"]                     # 段 VWAP（复权）
    cva, cvv = gd["va"].cumsum(), gd["vv"].cumsum()
    dvw = cva / cvv.where(cvv > 0) * g["adj"]
    out = pd.DataFrame({"iid": g["iid"], "day": g["day"], "seg": g["seg"]})
    out["SP_LSD"] = -(r_seg - r_day / 8)
    out["SP_LSDP"] = -(r_seg - r_day / nseg)
    out["SP_LSDV"] = -(np.log(svw / prev) - np.log(dvw / pca) / nseg)
    r_prev = gd["ca"].shift(1).pipe(lambda q: np.log(q / gd["ca"].shift(2).fillna(pca)))
    out["SP_ACC"] = -(r_seg - r_prev)
    out.loc[g["seg"] == "1000", "SP_ACC"] = np.nan
    out["SP_CLV"] = -((2 * g["c"] - g["h"] - g["l"]) / (g["h"] - g["l"]).where(g["h"] > g["l"])).clip(-1, 1)
    p10 = g["ca"].where(g["seg"] == "1000").groupby([g["iid"], g["day"]]).transform("max")
    out["SP_A1000"] = (-np.log(g["ca"] / p10)).where(g["seg"] > "1000")
    out["SP_RSEG"] = -r_seg                                                   # 原始成分（10-21 T10d）：本段反转、当天收益反向，让树自己组合
    out["SP_RDAY"] = -r_day
    out["_rd"] = r_day
    out = out[out["seg"] != "1500"].replace([np.inf, -np.inf], np.nan)
    out["MKT_DDISP"] = out.groupby(["day", "seg"])["_rd"].transform("std")      # 市况：当天至今收益的截面标准差（同一截面相同）
    return out


def main():
    with Pool(10) as p:
        parts = p.map(one, sorted(glob.glob(SRC)))
    A = features(pd.concat(parts, ignore_index=True))
    c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
    for nm in RANKED + RAW:
        Wd = A.pivot_table(index=["iid", "day"], columns="seg", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        x = H1.align(c, Wd)
        x = np.where(np.isfinite(x) & (x == 0), 1e-9, x); x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
