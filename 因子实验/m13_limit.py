# -*- coding: utf-8 -*-
"""方向 D（10-20）：A 股涨跌停机制。涨跌停价 = 昨收 ×（1 ± 幅度）四舍五入到分；幅度按板块：创业板（300／301）2020-08-24 起 20%、
科创板（688）20%、其余 10%（ST 无法识别，忽略）。截面 t 只用到 t 为止的 30 分钟 K 线（最高、最低、收盘）。
  LU_AT     截面收盘在涨停价（封板）            LD_AT     截面收盘在跌停价
  LU_BROKE  今日曾触及涨停、截面已不在涨停（炸板）  LD_BROKE  今日曾触及跌停、截面已离开（翘板）
  LU_DIST   距涨停的距离 =（涨停价 − 收盘）÷ 昨收   LD_DIST   距跌停的距离
  LU_Y／LD_Y 昨日收盘涨停／跌停（日内不变）
文献：涨跌停的磁吸效应（Du, Liu & Rhee 2009；Chen et al. 2019）、炸板后的回落（A 股券商研报）。
另打印事件研究：各事件的出现频率、事件股下一段标签秩的均值（相对 0）。保存 vc_tmp/h1_LU_*.npy、h1_LD_*.npy。"""
import glob
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = "/Users/user/Desktop/work/quant/数据/raw_bar30m/cpt_jyc_2026_e2e_bar30m/*.feather"
MAP = "/Users/user/Desktop/work/quant/fx/root/home/user/jyc_scorer/data/exposure/id_map.csv"
rows = []
for f in sorted(glob.glob(SRC)):
    d = pd.read_feather(f, columns=["date", "instrument_id", "high", "low", "close"])
    d["hm"] = d["date"].dt.strftime("%H%M"); d["day"] = d["date"].dt.normalize()
    rows.append(d[d["hm"].isin(H1.SEGS)][["instrument_id", "day", "hm", "high", "low", "close"]])
t = pd.concat(rows, ignore_index=True); del rows
mp = pd.read_csv(MAP, dtype=str)
code = dict(zip(mp["local_id"].str[2:].astype(int), mp["instrument"]))
Hh = t.pivot_table(index=["instrument_id", "day"], columns="hm", values="high", aggfunc="max").reindex(columns=H1.SEGS)
Ll = t.pivot_table(index=["instrument_id", "day"], columns="hm", values="low", aggfunc="min").reindex(columns=H1.SEGS)
Cc = t.pivot_table(index=["instrument_id", "day"], columns="hm", values="close", aggfunc="last").reindex(columns=H1.SEGS)
del t
iid = Cc.index.get_level_values(0).to_numpy(); day = Cc.index.get_level_values(1)
pre = pd.Series([code.get(int(i), "")[:3] for i in iid], index=Cc.index)
wide20 = (pre.isin(["300", "301"]) & (day >= pd.Timestamp("2020-08-24"))) | (pre == "688")
L = np.where(wide20, 0.20, 0.10)
c15 = Cc["1500"]
pc = c15.groupby(level=0, sort=False).shift(1).to_numpy(float)
up = np.floor(pc * (1 + L) + 0.5); dn = np.floor(pc * (1 - L) + 0.5)
runH = Hh.cummax(axis=1).to_numpy(float); runL = Ll.cummin(axis=1).to_numpy(float); C = Cc.to_numpy(float)
okp = np.isfinite(pc) & (pc > 0)
S = {}
S["LU_AT"] = (C >= up[:, None]) & okp[:, None]
S["LD_AT"] = (C <= dn[:, None]) & okp[:, None]
S["LU_BROKE"] = (runH >= up[:, None]) & ~S["LU_AT"] & okp[:, None]
S["LD_BROKE"] = (runL <= dn[:, None]) & ~S["LD_AT"] & okp[:, None]
dist_u = np.where(okp[:, None], (up[:, None] - C) / pc[:, None], np.nan)
dist_d = np.where(okp[:, None], (C - dn[:, None]) / pc[:, None], np.nan)
yu = pd.Series(S["LU_AT"][:, -1].astype(float), index=Cc.index).groupby(level=0, sort=False).shift(1).to_numpy()
yd = pd.Series(S["LD_AT"][:, -1].astype(float), index=Cc.index).groupby(level=0, sort=False).shift(1).to_numpy()
cs_idx = [H1.SEGS.index(s0) for s0 in H1.CS]
frames = {nm: pd.DataFrame(np.where(okp[:, None], S[nm][:, cs_idx].astype(float), np.nan), index=Cc.index, columns=H1.CS) for nm in S}
frames["LU_DIST"] = pd.DataFrame(dist_u[:, cs_idx], index=Cc.index, columns=H1.CS)
frames["LD_DIST"] = pd.DataFrame(dist_d[:, cs_idx], index=Cc.index, columns=H1.CS)
frames["LU_Y"] = pd.DataFrame({s0: yu for s0 in H1.CS}, index=Cc.index)
frames["LD_Y"] = pd.DataFrame({s0: yd for s0 in H1.CS}, index=Cc.index)
c = R.core(); z930 = c.slot_of_date[c.date_id] == 930
import d2_optimize as O
yr = O.rc_of(c.y, c.date_id, c.n_dates)
print("事件研究（截面行占比；事件股下一段标签秩均值，居中秩 −0.5～0.5）", flush=True)
for nm, fr in frames.items():
    v = H1.align(c, fr); v[z930] = np.nan
    if nm.endswith(("_AT", "_BROKE", "_Y")):
        m = (v == 1) & np.isfinite(yr)
        yrs = pd.to_datetime(c.key_date).year.to_numpy()
        per = " ".join("%d:%+.4f(%d)" % (y0, np.nanmean(yr[m & (yrs == y0)]), (m & (yrs == y0)).sum()) for y0 in range(2020, 2025))
        print("%-9s 占比 %.4f  标签秩均值 %+.4f  逐年 %s" % (nm, float(np.nanmean(v == 1)), float(np.nanmean(yr[m])), per), flush=True)
        out = np.where(v == 1, 1.0, np.where(np.isfinite(v), -1e-6, 0.0))          # 事件 = 1，非事件 = 极小负数（保持非零以免被当作缺失）
    else:
        out = np.nan_to_num(v, nan=0.0)
        out = np.where(out == 0, 1e-9, out) * np.isfinite(v)
    out[z930] = 0.0
    np.save(H1.OUT / ("h1_%s.npy" % nm), out)
    print("保存", nm, "非零率 %.3f" % float((out != 0).mean()), flush=True)
