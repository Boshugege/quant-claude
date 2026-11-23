# -*- coding: utf-8 -*-
"""第十七轮 L1：收盘行为族（全部当日不变、只用严格过去交易日）。

数据：store/platform/bar1m/YYYYMM.parquet（= 平台给因子脚本的 1 分钟表，与 1601／1232 同源）。
先按月聚成"日级原始量"缓存，再对每个原始量取严格过去 20 日均值（roll_prev → rolling + shift(1)）得到锚。

原始量（日级）：
  lab     = 收盘 ÷ 尾盘段(14:30,15:00] VWAP − 1            ← CLS20 的来源
  posHi   = 收盘 ÷ 当日最高 − 1                            ← CLOSEPOS20 的真实最高价版
  posLo   = 收盘 ÷ 当日最低 − 1
  rng     = (收盘−最低) ÷ (最高−最低)
  r5/r15/r30 = 收盘 ÷ 14:55／14:45／14:30 价 − 1
  auck    = 收盘 ÷ 14:57 价 − 1（收盘集合竞价价格冲击）
  auv     = 15:00 后成交量 ÷ 全天成交量（收盘竞价量占比）
  amtsh30 = 尾盘段成交额 ÷ 全天成交额
  amtsh5  = 最后 5 分钟成交额 ÷ 全天成交额
  dev     = 尾盘段收益 − 全天收益
"""
import argparse
import glob
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
import k1_anchor as K  # noqa: E402
from g_go import GO  # noqa: E402
from h1_season import roll_prev, CS  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
BARDIR = "/home/user/jyc_scorer/store/platform/bar1m/*.parquet"
CACHE = OUT / "l1_min_daily.parquet"
YRS = ["2023", "2024", "2021", "2022", "2020"]


def build_daily(force=False):
    if CACHE.exists() and not force:
        return pd.read_parquet(CACHE)
    parts = []
    for f in sorted(glob.glob(BARDIR)):
        d = pd.read_parquet(f, columns=["date", "instrument_id", "high", "low", "open", "close", "volume", "amount"])
        d["day"] = d["date"].dt.normalize()
        d["hm"] = (d["date"].dt.hour * 100 + d["date"].dt.minute).astype(np.int16)
        gi = ["instrument_id", "day"]
        g = d.groupby(gi, sort=False)
        a = g.agg(hi=("high", "max"), lo=("low", "min"), vol=("volume", "sum"),
                  amt=("amount", "sum"), o1=("open", "first"), c1500=("close", "last"))
        for tag, lim in (("c1430", 1430), ("c1445", 1445), ("c1455", 1455), ("c1457", 1457)):
            a[tag] = d[d["hm"] <= lim].groupby(gi, sort=False)["close"].last()
        t30 = d[(d["hm"] > 1430)]
        a["amt30"] = t30.groupby(gi, sort=False)["amount"].sum()
        a["vol30"] = t30.groupby(gi, sort=False)["volume"].sum()
        t5 = d[(d["hm"] > 1455)]
        a["amt5"] = t5.groupby(gi, sort=False)["amount"].sum()
        a["vauct"] = d[d["hm"] > 1457].groupby(gi, sort=False)["volume"].sum()
        parts.append(a.reset_index())
        print("  %s -> %d 行" % (pathlib.Path(f).name, len(a)), flush=True)
    t = pd.concat(parts, ignore_index=True)
    t.to_parquet(CACHE, index=False)
    print("缓存 %s：%d 行" % (CACHE, len(t)), flush=True)
    return t


def raw_series(t):
    t = t.copy()
    t["day"] = pd.to_datetime(t["day"])
    idx = pd.MultiIndex.from_arrays([t["instrument_id"].to_numpy(np.int64), t["day"].to_numpy()])
    S = lambda c: pd.Series(t[c].to_numpy(float), index=idx)  # noqa: E731
    c, hi, lo = S("c1500"), S("hi"), S("lo")
    amt, vol = S("amt"), S("vol")
    amt30, vol30 = S("amt30"), S("vol30")
    R = {}
    R["lab"] = c / (amt30 / vol30.where(vol30 > 0)) - 1.0
    R["posHi"] = c / hi - 1.0
    R["posLo"] = c / lo - 1.0
    R["rng"] = (c - lo) / (hi - lo).where(hi > lo)
    R["r5"] = c / S("c1455") - 1.0
    R["r15"] = c / S("c1445") - 1.0
    R["r30"] = c / S("c1430") - 1.0
    R["auck"] = c / S("c1457") - 1.0
    R["auv"] = S("vauct") / vol.where(vol > 0)
    R["amtsh30"] = amt30 / amt.where(amt > 0)
    R["amtsh5"] = S("amt5") / amt.where(amt > 0)
    R["dev"] = R["r30"] - (c / S("o1") - 1.0)
    return R


NAMES = {"lab": "LAB20", "posHi": "POSHI20", "posLo": "POSLO20", "rng": "RNG20",
         "r5": "R5_20", "r15": "R15_20", "r30": "R30_20", "auck": "AUCK20",
         "auv": "AUV20", "amtsh30": "AMTSH30_20", "amtsh5": "AMTSH5_20", "dev": "DEV20"}


def make_anchors(t, N=20):
    R = raw_series(t)
    return {NAMES[k]: roll_prev(v, N, max(N // 2, 5)) for k, v in R.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    t = build_daily(a.force)
    AN = make_anchors(t)
    del t
    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    y = np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)
    kd = pd.to_datetime(c.key_date)

    RSn, Xs = {}, {}
    for nm, s in AN.items():
        x, r = K.rank_of(c, s, z930)
        Xs[nm] = x
        RSn[nm] = r
        np.save(OUT / ("h1_%s.npy" % nm), x.astype(np.float32))
    AN2 = K.make_anchors(K.load_frames())   # 30 分钟口径的 CLS20 / CLOSEPOS20（K1／K2 用过的）
    for nm in ("CLS20", "CLOSEPOS20"):
        x, r = K.rank_of(c, AN2[nm], z930)
        Xs[nm] = x
        RSn[nm] = r
        np.save(OUT / ("h1_%s.npy" % nm), x.astype(np.float32))

    print("\n=== 单个特征：全截面 IC／14:30 截面 IC／日间持续性／与参照的相关 ===", flush=True)
    print("%-11s %s %8s %8s %8s %8s %8s" % (
        "特征", " ".join("%9s" % ("全IC" + w[-2:]) for w in YRS), "14:30均IC", "持续性",
        "vsCLS20", "vsCP20", "vs1363"), flush=True)
    ref = {"CLS20": RSn["CLS20"], "CLOSEPOS20": RSn["CLOSEPOS20"], "1363": G.B[("1363", 0)]}
    for nm in list(AN) + ["CLS20", "CLOSEPOS20"]:
        r = RSn[nm]
        full, ic1430 = [], []
        for w in YRS:
            m = (kd.year == int(w)) & ~z930
            dm = np.zeros(c.n_dates, bool)
            dm[np.unique(c.date_id[m])] = True
            full.append(c.per_date_rank_corr(r, y, dm))
            dm2 = np.zeros(c.n_dates, bool)
            m2 = m & (c.slot_of_date[c.date_id] == 1430)
            dm2[np.unique(c.date_id[m2])] = True
            ic1430.append(c.per_date_rank_corr(r, y, dm2))
        # 日间持续性（10:00 截面）
        x = np.asarray(Xs[nm], float)
        m0 = np.isfinite(x) & (c.slot_of_date[c.date_id] == 1000)
        idx0 = np.flatnonzero(m0)
        rr = O.rc_of(np.where(m0, x, np.nan), c.date_id, c.n_dates)[idx0]
        w0 = pd.DataFrame({"d": c.date_id[idx0], "i": c.inst[idx0], "r": rr}).pivot_table(
            index="d", columns="i", values="r", aggfunc="first")
        per = float(w0.corrwith(w0.shift(1), axis=1).mean())
        dm = np.zeros(c.n_dates, bool)
        dm[np.unique(c.date_id[(kd.year.isin((2023, 2024))) & ~z930])] = True
        cr = {k: c.per_date_rank_corr(x, np.nan_to_num(v, nan=0.0), dm) for k, v in ref.items()}
        print("%-11s %s %+8.5f %8.3f %+8.3f %+8.3f %+8.3f" % (
            nm, " ".join("%+9.5f" % v for v in full), float(np.nanmean(ic1430)), per,
            cr["CLS20"], cr["CLOSEPOS20"], cr["1363"]), flush=True)
    print("\nnpy 已保存到 %s/h1_<特征>.npy（对齐到评估行、09:30 置 0）" % OUT, flush=True)


if __name__ == "__main__":
    main()
