# -*- coding: utf-8 -*-
"""第十四轮 H1：同时段季节性做成一个"族"（新信息源）。

数据：data/bar30m/*.feather（30 分钟 bar，含 09:25 竞价 bar；价格是原始价，日内比值不受复权影响）
      与 1232 平台的 1 分钟聚段口径一致：段 (09:30,10:00]…(14:30,15:00] 直接由 30m bar 给出，不用跑 SQL。

族成员（全部严格只用截面之前的数据）：
  L_N    N=5/20/40/60：下一个半小时段在过去 N 日的 lab 均值（lab = 段收盘/段VWAP - 1，= 1232 口径）
  Lret20 下一个半小时段在过去 20 日的段收益均值
  VSH20  下一个半小时段在过去 20 日的成交额占全天比重均值
  PACE20 今日累计成交额 / 过去 20 日同一时刻累计成交额均值
  AC_N   N=20/40：过去 N 日（本段, 下段）收益的回归斜率（>0 延续，<0 反转）
  DOW10  过去 10 个同星期几的下一段 lab 均值
"""
import argparse
import glob
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
CACHE = OUT / "h1_seg.parquet"
BARDIR = "/home/user/jyc_scorer/data/bar30m/*.feather"
AUCT = "0925"
SEGS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
CS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430"]
NEXT = {"1000": "1030", "1030": "1100", "1100": "1130", "1130": "1330",
        "1330": "1400", "1400": "1430", "1430": "1500"}
SLOT = {s: int(s) for s in CS}


def build_cache():
    rows = []
    for f in sorted(glob.glob(BARDIR)):
        d = pd.read_feather(f, columns=["date", "instrument_id", "open", "close", "volume", "amount"])
        d["hm"] = d["date"].dt.strftime("%H%M")
        d["day"] = d["date"].dt.normalize()
        rows.append(d[["day", "instrument_id", "hm", "open", "close", "volume", "amount"]])
    t = pd.concat(rows, ignore_index=True)
    del rows
    t = t[t["hm"].isin([AUCT] + SEGS)]
    t = t.groupby(["day", "instrument_id", "hm"], as_index=False).agg(
        open=("open", "first"), close=("close", "last"),
        volume=("volume", "sum"), amount=("amount", "sum"))
    t.to_parquet(CACHE, index=False)
    print("缓存 %s 行 %d" % (CACHE, len(t)))


def wide(t):
    W = {}
    for f in ("open", "close", "volume", "amount"):
        W[f] = t.pivot_table(index=["instrument_id", "day"], columns="hm", values=f, aggfunc="last")
        W[f] = W[f].reindex(columns=[AUCT] + SEGS)
    return W


def roll_prev(s, N, minp, how="mean"):
    """同一股票按日滚动，取严格过去 N 日（再 shift 1）。s 为 Series 或 DataFrame，索引 (iid, day)。"""
    g = s.groupby(level=0, sort=False)
    f = {"mean": "mean", "sum": "sum"}[how]
    m = getattr(g.rolling(N, min_periods=minp), f)()
    m.index = m.index.droplevel(0)
    m = m.reindex(s.index)
    return m.groupby(level=0, sort=False).shift(1)


def make_signals(c):
    t = pd.read_parquet(CACHE)
    t["day"] = pd.to_datetime(t["day"])
    W = wide(t)
    del t
    close, open_, amt, vol = W["close"], W["open"], W["amount"], W["volume"]
    idx = close.index
    iid = idx.get_level_values(0)
    day = idx.get_level_values(1)
    wd = pd.Series(pd.DatetimeIndex(day).weekday, index=idx)
    vwap = amt / vol.where(vol > 0)
    lab = close[SEGS] / vwap[SEGS] - 1.0
    prev = close[SEGS].shift(1, axis=1)
    prev["1000"] = open_["1000"]
    ret = close[SEGS] / prev - 1.0
    daytot = amt[SEGS].sum(axis=1)
    vsh = amt[SEGS].div(daytot.where(daytot > 0), axis=0)
    cum = amt[SEGS].cumsum(axis=1)

    sig = {}

    def put(name, frame):
        f = frame[CS]
        f = f.rename_axis(index=["iid", "day"], columns="slot")
        sig[name] = f

    # A. 多回看
    for N in (5, 20, 40, 60):
        mu = roll_prev(lab[list(NEXT.values())], N, max(N // 2, 3))
        mu.columns = CS
        put("L_%d" % N, mu)
    mu = roll_prev(ret[list(NEXT.values())], 20, 10)
    mu.columns = CS
    put("Lret20", mu)
    # B. 成交量季节性 + 今日放量
    mu = roll_prev(vsh[list(NEXT.values())], 20, 10)
    mu.columns = CS
    put("VSH20", mu)
    prevcum = roll_prev(cum[CS], 20, 10)
    put("PACE20", cum[CS] / prevcum.where(prevcum > 0))
    # C. 相邻窗口关系（回归斜率）
    for N in (20, 40):
        xy = pd.DataFrame({s0: ret[s0] * ret[NEXT[s0]] for s0 in CS})
        xx = pd.DataFrame({s0: ret[s0] * ret[s0] for s0 in CS})
        num = roll_prev(xy, N, N // 2, "sum")
        den = roll_prev(xx, N, N // 2, "sum")
        put("AC_%d" % N, num / den.where(den > 0))
    # D. 星期几（过去 10 个同星期几的下一段 lab 均值）
    dow = lab[[NEXT[s0] for s0 in CS]]
    dow.columns = CS
    d = dow.copy()
    d["_iid"] = d.index.get_level_values(0)
    d["_day"] = d.index.get_level_values(1)
    d["_wd"] = pd.DatetimeIndex(d["_day"]).weekday
    d = d.sort_values(["_iid", "_wd", "_day"])
    d[CS] = d.groupby(["_iid", "_wd"], sort=False)[CS].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    d = d.set_index(["_iid", "_day"]).sort_index()[CS]
    put("DOW10", d)
    return sig


def align(c, frame):
    n = c.n_rows
    o = np.full(n, np.nan)
    day = pd.to_datetime(c.day_of_date[c.date_id])
    need = pd.DataFrame({"row": np.arange(n, dtype=np.int64), "iid": c.inst.astype(np.int64),
                         "day": day.values, "slot": c.slot_of_date[c.date_id]})
    for s0 in CS:
        sub = frame[s0]
        d = pd.DataFrame({"iid": sub.index.get_level_values(0).to_numpy(),
                          "day": sub.index.get_level_values(1).to_numpy(),
                          "v": sub.to_numpy(float)})
        q = need[need["slot"] == SLOT[s0]]
        m = q.merge(d, on=["iid", "day"], how="left", sort=False)
        o[m["row"].to_numpy()] = m["v"].to_numpy(float)
    return o


WW = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
      ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
      ("2020", "2020-01-01", "2020-12-31")]


def en_of(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="single")
    ap.add_argument("--names", default="")
    a = ap.parse_args()
    if not CACHE.exists():
        build_cache()
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    if a.mode == "cache":
        return
    if a.mode != "eval":
        SIG = make_signals(c)
        print("族成员：%s" % list(SIG))
        names = a.names.split(",") if a.names else list(SIG)
        for nm in names:
            x = align(c, SIG[nm])
            x[z930] = 0.0
            np.save(OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("已保存 npy")
        if a.mode == "save":
            return
    names = a.names.split(",") if a.names else sorted(p.name[3:-4] for p in OUT.glob("h1_*.npy"))
    # 评估
    h1232 = R.load("fm_1232")[0]
    h1232 = np.nan_to_num(np.asarray(h1232, float), nan=0.0)
    s161 = np.nan_to_num(O.axis(c, "fm_s161", "2020-01-01", "2024-12-31"), nan=0.0)
    msk = np.isfinite(c.y) & (~z930) & (pd.to_datetime(c.key_date).year == 2023)
    for nm in names:
        x = np.load(OUT / ("h1_%s.npy" % nm))
        rx = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
        c1232 = np.nanmean(c.per_date_rank_corr(rx, h1232, msk))
        c161 = np.nanmean(c.per_date_rank_corr(rx, s161, msk))
        line = "%-8s |1232 %+.3f |161 %+.3f" % (nm, c1232, c161)
        for wn, st, en in WW:
            m, _ = c.evaluate(np.where(z930, 0.0, x), start=st, end=en_of(en))
            line += " |%s IC %.5f IR %.4f 换手 %.4f" % (wn, m["ic_mean"], m["ic_ir"], m["turnover"])
        print(line)


if __name__ == "__main__":
    main()
