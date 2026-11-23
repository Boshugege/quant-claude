# -*- coding: utf-8 -*-
"""因子头脑风暴（10-18 晚）第一步：从 30 分钟 bar（OHLCV、成交笔数、1–3 档盘口快照）一次性构造 24 个候选根，对齐到 engine 面板，存 OUT/bs_<名>.npy。
截面 t 只用"段末时刻 ≤ t"的 bar（段末快照在 t 时刻已知）；跨日特征只用之前的交易日；09:30 截面由评估统一置 0。"""
import glob
import numpy as np
import pandas as pd
from g_go import GO
from c1_upgrades import OUT
G = GO(); c = G.c
cols = ["date", "instrument_id", "open", "high", "low", "close", "volume", "amount", "deal_number"] + \
       ["%s_%s%d" % (s, k, i) for s in ("bid", "ask") for k in ("price", "volume", "num_orders") for i in (1, 2, 3)]
D = pd.concat([pd.read_feather(f, columns=cols) for f in sorted(glob.glob(str(OUT).replace("srv/media/jyc_scorer/vc_tmp", "home/user/jyc_scorer/data/bar30m") + "/*.feather"))], ignore_index=True)
D = D.rename(columns={"instrument_id": "iid"})
D["day"] = D["date"].dt.normalize(); D["hm"] = D["date"].dt.strftime("%H%M")
D = D.sort_values(["iid", "date"]).reset_index(drop=True)
for k in ("open", "high", "low", "close", "volume", "amount", "deal_number"):
    D[k] = D[k].astype(float)
g = D.groupby(["iid", "day"], sort=False)
eps = 1e-12
with np.errstate(all="ignore"):
    D["prev_close_bar"] = g["close"].shift(1)
    D["r_last"] = D["close"] / D["prev_close_bar"] - 1
    D["cvwap"] = D["close"] / (D["amount"] / D["volume"].where(D["volume"] > 0)) - 1
    dc = D[D["hm"] == "1500"].set_index(["iid", "day"])["close"]
    dvw = D.groupby(["iid", "day"])["amount"].sum() / D.groupby(["iid", "day"])["volume"].sum().replace(0, np.nan)
    tail = (D[D["hm"] == "1500"].set_index(["iid", "day"])["close"] / D[D["hm"] == "1430"].set_index(["iid", "day"])["close"] - 1)
    dd = pd.DataFrame({"dclose": dc, "dvwap": dvw, "dtail": tail}).sort_index()
    gd = dd.groupby(level=0)
    dd["pc"] = gd["dclose"].shift(1); dd["p_cvw"] = (dd["dclose"] / dd["dvwap"] - 1).groupby(level=0).shift(1); dd["p_tail"] = gd["dtail"].shift(1)
    D = D.join(dd[["pc", "p_cvw", "p_tail"]], on=["iid", "day"])
    op = D[D["hm"] == "0925"].set_index(["iid", "day"])["close"].rename("op0925")
    D = D.join(op, on=["iid", "day"])
    g = D.groupby(["iid", "day"], sort=False)
    D["r_day"] = D["close"] / D["pc"] - 1
    D["r_intra"] = D["close"] / D["op0925"] - 1
    D["gap"] = D["op0925"] / D["pc"] - 1
    D["hi_sofar"] = g["high"].cummax(); D["lo_sofar"] = g["low"].cummin()
    D["pos"] = (D["close"] - D["lo_sofar"]) / (D["hi_sofar"] - D["lo_sofar"]).where(D["hi_sofar"] > D["lo_sofar"])
    bv1, av1 = D["bid_volume1"].astype(float), D["ask_volume1"].astype(float)
    bv3 = D[["bid_volume1", "bid_volume2", "bid_volume3"]].sum(1).astype(float); av3 = D[["ask_volume1", "ask_volume2", "ask_volume3"]].sum(1).astype(float)
    bn3 = D[["bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]].sum(1).astype(float); an3 = D[["ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]].sum(1).astype(float)
    D["obi1"] = (bv1 - av1) / (bv1 + av1).where((bv1 + av1) > 0)
    D["obi3"] = (bv3 - av3) / (bv3 + av3).where((bv3 + av3) > 0)
    a1, b1 = D["ask_price1"].astype(float), D["bid_price1"].astype(float)
    D["spread"] = ((a1 - b1) / ((a1 + b1) / 2)).where((a1 > 0) & (b1 > 0))
    D["ocnt"] = (bn3 - an3) / (bn3 + an3).where((bn3 + an3) > 0)
    D["osize"] = np.log((bv3 / bn3.where(bn3 > 0)) / (av3 / an3.where(an3 > 0)))
    bd = sum(D["bid_price%d" % i].astype(float) * D["bid_volume%d" % i].astype(float) for i in (1, 2, 3))
    ad = sum(D["ask_price%d" % i].astype(float) * D["ask_volume%d" % i].astype(float) for i in (1, 2, 3))
    D["depth"] = np.log(bd.where(bd > 0) / ad.where(ad > 0))
    D["obi_chg"] = D["obi1"] - D.groupby(["iid", "day"], sort=False)["obi1"].shift(1)
    ts = D["amount"] / D["deal_number"].where(D["deal_number"] > 0)
    D["tsize_rel"] = np.log(ts / (g["amount"].cumsum() / g["deal_number"].cumsum().replace(0, np.nan)))
    D["ushadow"] = ((D["high"] - D[["open", "close"]].max(1)) / (D["high"] - D["low"]).where(D["high"] > D["low"]))
    D["lshadow"] = ((D[["open", "close"]].min(1) - D["low"]) / (D["high"] - D["low"]).where(D["high"] > D["low"]))
    D["absr_amt"] = D["r_last"].abs() / D["amount"].where(D["amount"] > 0) * 1e8
    g = D.groupby(["iid", "day"], sort=False)
    D["amihud"] = np.log(g["absr_amt"].transform(lambda s: s.expanding().mean()) + eps)
    D["absr"] = D["r_last"].abs()
    xa, ya = D["absr"].fillna(0.0), D["volume"].where(D["absr"].notna(), 0.0)
    D["_n"] = D["absr"].notna().astype(float); D["_x"] = xa; D["_y"] = ya; D["_xy"] = xa * ya; D["_xx"] = xa * xa; D["_yy"] = ya * ya
    cs = D.groupby(["iid", "day"], sort=False)[["_n", "_x", "_y", "_xy", "_xx", "_yy"]].cumsum()
    n = cs["_n"]
    cov = cs["_xy"] / n - (cs["_x"] / n) * (cs["_y"] / n)
    vx = cs["_xx"] / n - (cs["_x"] / n) ** 2; vy = cs["_yy"] / n - (cs["_y"] / n) ** 2
    D["pv_corr"] = (cov / np.sqrt(vx * vy).where((vx > 0) & (vy > 0))).where(n >= 3)    # 当日到 t 为止各段 |r| 与成交量的相关
    gs = D.groupby(["iid", "hm"], sort=False)
    D["v_sameslot20"] = gs["volume"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    D["vol_surge"] = np.log(D["volume"] / D["v_sameslot20"].where(D["v_sameslot20"] > 0))
    D["rng"] = (D["high"] - D["low"]) / D["close"]
    D["rng20"] = gs["rng"].transform(lambda s: s.rolling(20, min_periods=10).mean().shift(1))
    D["rng_rel"] = np.log(D["rng"] / D["rng20"].where(D["rng20"] > 0))
    D["rev_x_vol"] = -D["r_last"] * D["vol_surge"]
    D["r_mkt"] = D["r_day"] - D.groupby("date")["r_day"].transform("mean")
FEATS = ["r_last", "cvwap", "r_day", "r_intra", "pos", "obi1", "obi3", "spread", "ocnt", "osize", "depth", "obi_chg", "tsize_rel",
         "ushadow", "lshadow", "amihud", "pv_corr", "vol_surge", "rng_rel", "rev_x_vol", "r_mkt", "gap", "p_tail", "p_cvw"]
# 对齐到面板：截面 t 的行用"段末 = t"的 bar；09:30 截面用 09:25 竞价 bar
D["sec"] = D["date"].where(D["hm"] != "0925", D["date"].dt.normalize() + pd.Timedelta("9h30min"))
need = pd.DataFrame({"row": np.arange(c.n_rows), "iid": c.inst.astype(np.int64), "sec": pd.to_datetime(c.key_date)})
M = need.merge(D[["iid", "sec"] + FEATS], on=["iid", "sec"], how="left").sort_values("row")
for f in FEATS:
    x = np.array(M[f].to_numpy(float), dtype=float, copy=True); x[~np.isfinite(x)] = np.nan
    np.save(OUT / ("bs_%s.npy" % f), x)
    print("%-10s 非空率 %.3f" % (f, np.isfinite(x).mean()), flush=True)
print("done", flush=True)
