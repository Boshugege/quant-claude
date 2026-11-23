# -*- coding: utf-8 -*-
"""G5：真实换手率家族（成交额 ÷ 前一交易日流通市值）——新信息源。

数据：data/bar30m/*.feather（30 分钟 bar，含 09:25 竞价 bar；amount 单位"分·股"）；
前一日流通市值来自暴露表 float_market_cap（严格前一交易日，合法）。

输出对齐到 engine.Core 行的数组（09:30 置 0），存 vc_tmp/g5_*.npy：
  TNOW  当日截至该截面的累计成交额 / 前一交易日流通市值
  TPREV 前一交易日全天成交额 / 前一交易日流通市值（日内常数 → 换手≈0）
  TREL  TNOW / 过去 20 个交易日的日均换手率
  AUCK  09:25 竞价成交额 / 当日截至该截面累计成交额
"""
import glob
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402

BAR = "/home/user/jyc_scorer/data/bar30m/*.feather"
EXP = "/home/user/jyc_scorer/data/exposure/exposure_%d.parquet"
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
# core 截面 → 累计到哪个 bar；09:30 无（置 0）
SEC2BAR = {930: None, 1000: "10:00", 1030: "10:30", 1100: "11:00", 1130: "11:30",
           1330: "13:30", 1400: "14:00", 1430: "14:30"}


def main():
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930

    # ---- 1) 30 分钟 bar → (day, inst) 的累计成交额
    rows = []
    for f in sorted(glob.glob(BAR)):
        d = pd.read_feather(f, columns=["date", "instrument_id", "amount"])
        d["hm"] = d["date"].dt.strftime("%H:%M")
        d["day"] = d["date"].dt.normalize()
        rows.append(d.groupby(["day", "instrument_id", "hm"], as_index=False)["amount"].sum())
        del d
    bars = pd.concat(rows, ignore_index=True)
    del rows
    piv = bars.pivot_table(index=["day", "instrument_id"], columns="hm", values="amount",
                           aggfunc="sum").fillna(0.0)
    del bars
    order = ["09:25", "10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "15:00"]
    piv = piv.reindex(columns=order, fill_value=0.0)
    cum = piv.cumsum(axis=1)
    cum.columns = ["amt_" + x.replace(":", "") for x in order]
    tab = cum.reset_index()
    del piv, cum

    # ---- 2) 前一日流通市值
    caps = [pd.read_parquet(EXP % y, columns=["date", "instrument", "float_market_cap"])
            for y in (2020, 2021, 2022, 2023, 2024)]
    cap = pd.concat(caps, ignore_index=True)
    imap = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    imap["iid"] = imap["local_id"].str.replace("ID", "", regex=False).astype(int)
    code2id = dict(zip(imap["instrument"].to_numpy(), imap["iid"].to_numpy()))
    cap["instrument_id"] = cap["instrument"].map(code2id)
    cap = cap.dropna(subset=["instrument_id"])
    cap["instrument_id"] = cap["instrument_id"].astype(int)
    cap["day"] = pd.to_datetime(cap["date"]).dt.normalize()
    cap = cap[["day", "instrument_id", "float_market_cap"]].drop_duplicates(["day", "instrument_id"], keep="last")
    cap = cap.sort_values(["instrument_id", "day"])
    cap["cap_prev"] = cap.groupby("instrument_id")["float_market_cap"].shift(1)
    cap = cap[["day", "instrument_id", "cap_prev"]]

    tab = tab.merge(cap, on=["day", "instrument_id"], how="left")
    tab = tab.sort_values(["instrument_id", "day"])
    g = tab.groupby("instrument_id")
    tab["turn_day"] = tab["amt_1500"] / tab["cap_prev"]
    tab["turn_prev"] = g["turn_day"].shift(1)
    tab["turn_ma20"] = tab.groupby("instrument_id")["turn_prev"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    print("bar/换手表行 %d" % len(tab))

    # ---- 3) 对齐到 core 行（merge，避免重复索引）
    need = pd.DataFrame({"row": np.arange(c.n_rows),
                         "day": pd.to_datetime(c.day_of_date[c.date_id]),
                         "instrument_id": c.inst.astype(int)})
    m = need.merge(tab, on=["day", "instrument_id"], how="left", sort=False)
    row = m["row"].to_numpy()
    n = c.n_rows

    def vec(col):
        o = np.full(n, np.nan)
        o[row] = m[col].to_numpy(float)
        return o

    amt = {s: (vec("amt_" + SEC2BAR[s].replace(":", "")) if SEC2BAR[s] else np.zeros(n)) for s in SEC2BAR}
    cap_prev = vec("cap_prev")
    TNOW = np.full(n, np.nan)
    for s in SEC2BAR:
        mm = c.slot_of_date[c.date_id] == s
        with np.errstate(invalid="ignore", divide="ignore"):
            TNOW[mm] = np.where(cap_prev[mm] > 0, amt[s][mm] / cap_prev[mm], np.nan)
    tp = vec("turn_prev")
    tm = vec("turn_ma20")
    with np.errstate(invalid="ignore", divide="ignore"):
        TREL = np.where(tm > 0, TNOW / tm, np.nan)
    auck = vec("amt_0925")
    AUCK = np.full(n, np.nan)
    for s in SEC2BAR:
        if SEC2BAR[s] is None:
            continue
        mm = c.slot_of_date[c.date_id] == s
        with np.errstate(invalid="ignore", divide="ignore"):
            AUCK[mm] = np.where(amt[s][mm] > 0, auck[mm] / amt[s][mm], np.nan)

    for nm, arr in (("TNOW", TNOW), ("TPREV", tp), ("TREL", TREL), ("AUCK", AUCK)):
        a = np.asarray(arr, float).copy()
        a[z930] = 0.0
        np.save(OUT / ("g5_%s.npy" % nm), a)
        fin = np.isfinite(a) & ~z930
        print("%-6s 有效 %d  中位 %.4g  p1 %.4g p99 %.4g" % (nm, fin.sum(), np.nanmedian(a[fin]),
                                                             np.nanpercentile(a[fin], 1), np.nanpercentile(a[fin], 99)))


if __name__ == "__main__":
    main()
