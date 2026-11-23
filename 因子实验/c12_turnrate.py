# -*- coding: utf-8 -*-
"""Claude C12：暴露表口径的真实换手率做"日内不变锚"，叠到 1609（第四类信息源，用户点名；严格错位）。
日换手 tr(d) = d 日 09:30–15:00 成交额 ÷ (d−1) 日流通市值（暴露表 float_market_cap，按股票 shift 1）。
锚（截面日 t 只用到 t−1 及更早）：TURN20 = tr 的 20 日均值；TURN5；TREL = log(TURN5 ÷ TURN60)。全天同值，逐截面秩。
G5（§112）在 1363 上测过 TNOW／TPREV／TREL：叠加增量 ≤0.0005。这里换到 1609 底座，并看与 CP 的配对。
用法（personal）：PYTHONPATH=因子实验 ~/jyc_scorer/.venv/bin/python 因子实验/c12_turnrate.py
"""
import glob
import numpy as np
import pandas as pd
import root_eval as R
import d2_optimize as O
from c1_upgrades import G, CP, CLS, ax, W1363, SMALL, DEAL, MET
from g_go import WINS, IXW

c = G.c; z930 = G.z930; slot = c.slot_of_date[c.date_id]
SEGS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
parts = []
for f in sorted(glob.glob("/home/user/jyc_scorer/data/bar30m/*.feather")):
    t = pd.read_feather(f, columns=["date", "instrument_id", "amount"])
    t = t[t["date"].dt.strftime("%H%M").isin(SEGS)]
    parts.append(t.assign(day=t["date"].dt.normalize()).groupby(["instrument_id", "day"])["amount"].sum())
A = pd.concat(parts).rename("amt").reset_index().rename(columns={"instrument_id": "iid"})
idm = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv")
idm["iid"] = idm["local_id"].str[2:].astype(int)
E = pd.concat([pd.read_parquet(f, columns=["date", "instrument", "float_market_cap"])
               for f in sorted(glob.glob("/home/user/jyc_scorer/data/exposure/exposure_20*.parquet"))])
E = E.merge(idm[["instrument", "iid"]], on="instrument").sort_values(["iid", "date"])
E["mcap_prev"] = E.groupby("iid")["float_market_cap"].shift(1)
D = A.merge(E[["iid", "date", "mcap_prev"]].rename(columns={"date": "day"}), on=["iid", "day"], how="left").sort_values(["iid", "day"])
D["tr"] = D["amt"] / D["mcap_prev"].where(D["mcap_prev"] > 0)
g = D.groupby("iid", sort=False)["tr"]
for N in (5, 20, 60):
    D["m%d" % N] = g.transform(lambda s: s.rolling(N, min_periods=max(N // 2, 3)).mean().shift(1))
D["TURN20"] = D["m20"]; D["TURN5"] = D["m5"]
with np.errstate(all="ignore"):
    D["TREL"] = np.log(D["m5"] / D["m60"])
print("日表", len(D), "缺失率", D[["TURN20", "TREL"]].isna().mean().round(3).to_dict(), flush=True)
need = pd.DataFrame({"row": np.arange(c.n_rows), "iid": c.inst.astype(np.int64), "day": pd.to_datetime(c.day_of_date[c.date_id]).values})
M = need.merge(D[["iid", "day", "TURN20", "TURN5", "TREL"]], on=["iid", "day"], how="left").sort_values("row")
RK = {}
for k in ("TURN20", "TURN5", "TREL"):
    r = np.nan_to_num(O.rc_of(M[k].to_numpy(float), c.date_id, c.n_dates), nan=0.0)
    r[z930] = 0.0
    RK[k] = r
NAMES = ("fm_s152", "o21_s152", "o20_s152")
K = 0.2908 / 0.289
SAFE = [r for r in W1363 if r not in DEAL]


def base(ix):
    B = np.asarray(R.load(NAMES[ix])[0], float).copy(); B[z930] = 0.0
    for r in SAFE:
        B += W1363[r] * ax(r, ix)
    for k, w in SMALL.items():
        B += w * G.axis(k, ix)
    B += 0.10 * K * CP + 0.05 * K * CLS + 0.10 * K * CLS * (slot == 1430)
    B[z930] = 0.0
    return B


def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn, _, _ in WINS}


BS = {ix: base(ix) for ix in (0, 1, 2)}
ref = ev5(BS)
print("1609 参考 " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in ("2023", "2024", "2021", "2022", "2020")), flush=True)
print("== 单独（全截面）IC／IC_IR／换手", flush=True)
for k, r in RK.items():
    res = ev5({0: r, 1: r, 2: r})
    print("%-7s %s" % (k, " | ".join("%s %+.4f/%+.3f/%.3f" % (wn, res[wn]["ic_mean"], res[wn]["ic_ir"], res[wn]["turnover"]) for wn in ("2023", "2024", "2021", "2022", "2020"))), flush=True)
print("== 叠到 1609：Δ IC／IC_IR／多空／压力／换手", flush=True)
for k, r in RK.items():
    for w in (-0.05, -0.03, 0.03, 0.05):
        res = ev5({ix: BS[ix] + w * K * r for ix in (0, 1, 2)})
        line, star = "%-7s %+.2f" % (k, w), 0
        for wn in ("2023", "2024", "2021", "2022", "2020"):
            d = {m: res[wn][m] - ref[wn][m] for m in MET}
            ok = d["ic_mean"] >= 0 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            star += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % star, flush=True)
