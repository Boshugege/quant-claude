# -*- coding: utf-8 -*-
"""D1 第二步：同伴组换成 sushirui 的相关性网络（不依赖申万行业）。

网络构造复用 /tmp/s/s152.py 的口径：每个自然月月初，用**此前 60 个交易日**的日收益算相关，
每只股票取相关性最高的 30 只为同伴（至少 30 个共同观测；只用过去数据）。

日收益来自本地暴露表 /home/user/jyc_scorer/data/exposure/exposure_*.parquet 的 ret 列
（该列是当日收益，按股票 shift(1) 只用前一交易日及更早，天然因果）。

用法：
  build   : python 因子实验/d1_peer_network.py --build
  rank    : ... --rank fm_1232 --tag 1232 --submit
  replace : ... --rank fm_1232 --tag P1232 --replace --h 1015-203626-f5f0 --w 0.20 --ewma 0.75 --submit
"""
import argparse
import glob
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_values_composite import read_raw, pct_rank  # noqa: E402

EXP = sorted(glob.glob("/home/user/jyc_scorer/data/exposure/exposure_*.parquet"))
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
PEER = OUT / "d1_peers.npz"
URL = "http://127.0.0.1:46220/api/jobs"
LOOKBACK, PEER_K, MIN_OBS = 60, 30, 30


def daily_ret():
    parts = [pd.read_parquet(f, columns=["date", "instrument", "ret"]) for f in EXP]
    d = pd.concat(parts, ignore_index=True)
    d["day"] = pd.to_datetime(d["date"]).dt.normalize()
    d["instrument"] = d["instrument"].astype(str)
    w = d.pivot_table(index="day", columns="instrument", values="ret", aggfunc="last").sort_index()
    return w


def build():
    ret = daily_ret()
    print("日收益 %s" % (ret.shape,))
    insts = ret.columns.values
    days = ret.index
    month = days.to_period("M")
    keep = {}
    for m in sorted(set(month)):
        in_month = days[month == m]
        prior_all = days[days < in_month[0]]
        if len(prior_all) < MIN_OBS:
            continue
        prior = prior_all[-LOOKBACK:]
        hist = ret.loc[prior]
        ok = (hist.notna().sum() >= MIN_OBS).values
        sub = np.where(ok)[0]
        if len(sub) < 2:
            continue
        corr = hist.iloc[:, sub].corr(min_periods=MIN_OBS).values
        np.fill_diagonal(corr, -np.inf)
        corr = np.nan_to_num(corr, nan=-np.inf)
        k = min(PEER_K, max(1, len(sub) - 1))
        idx = np.full((len(insts), PEER_K), -1, dtype=np.int32)
        order = np.argsort(-corr, axis=1, kind="stable")[:, :k]
        idx[sub] = sub[order]
        keep[str(m)] = idx
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PEER, insts=insts.astype(str), months=np.array(sorted(keep)),
                        **{k: v.astype(np.int32) for k, v in keep.items()})
    print("写出 %s 月份 %d 只股票 %d" % (PEER, len(keep), len(insts)))


def load_peers():
    z = np.load(PEER, allow_pickle=False)
    insts = z["insts"]
    return insts, {m: z[m] for m in z["months"]}


def peer_rank(df, insts, peers):
    """df: date/instrument/factor → 同伴内百分位排名 − 0.5（同伴有效值 < 3 时回退截面排名）。"""
    df = df.copy()
    df["day"] = pd.to_datetime(df["date"]).dt.normalize()
    df["mon"] = df["day"].dt.to_period("M").astype(str)
    pos = pd.Index(insts).get_indexer(df["instrument"].astype(str).values)
    out = np.full(len(df), np.nan)
    for m, rows in df.groupby("mon", sort=False).indices.items():
        idx = peers.get(m)
        sub = df.iloc[rows]
        if idx is None:
            continue
        wide = sub.pivot_table(index="date", columns="instrument", values="factor", aggfunc="last")
        wide = wide.reindex(columns=insts)
        x = wide.values.astype(np.float32)
        valid = idx >= 0
        safe = np.where(valid, idx, 0)
        g = x[:, safe]                                  # (T, N, K)
        g = np.where(valid[None, :, :], g, np.nan)
        own = x[:, :, None]
        ok = ~np.isnan(g) & ~np.isnan(own)
        cnt = ok.sum(axis=2)
        score = ((g < own) & ok).sum(axis=2) + 0.5 * ((g == own) & ok).sum(axis=2)
        rel = np.where(cnt >= 3, score / np.maximum(cnt, 1) - 0.5, np.nan).astype(np.float64)
        r_pos = wide.index.get_indexer(sub["date"])
        out[rows] = rel[r_pos, np.maximum(pos[rows], 0)]
    df["out"] = out
    fb = df.groupby("date")["factor"].rank(pct=True) - 0.5
    df["out"] = df["out"].fillna(fb).fillna(0.0)
    df.loc[pd.to_datetime(df["date"]).dt.strftime("%H%M") == "0930", "out"] = 0.0
    return df[["date", "instrument"]].assign(factor=df["out"].to_numpy(float)).sort_values(
        ["date", "instrument"]).reset_index(drop=True)


def submit(parquet, name, start, end, baseline):
    cmd = ["curl", "-s", "-F", "files=@%s" % parquet, "-F", "kind=values",
           "-F", "start=%s" % start, "-F", "end=%s" % end,
           "-F", "name=%s" % name, "-F", "baseline=%s" % baseline, URL]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("提交 %s -> %s" % (name, r.stdout.strip() or r.stderr.strip()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--rank", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--baseline", default="1015-203626-f5f0")
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--h", default="1015-203626-f5f0")
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--ewma", type=float, default=0.0)
    ap.add_argument("--submit", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
        return
    if not a.rank:
        raise SystemExit("要么 --build，要么 --rank SRC")
    insts, peers = load_peers()
    _xd, xdf = read_raw(a.rank)
    pr = peer_rank(xdf, insts, peers)
    out = pathlib.Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    name = "pn_%s" % a.tag
    f = out / ("v_%s.parquet" % name)
    pr.to_parquet(f, index=False)
    print("写出 %s rows=%d" % (f, len(pr)))
    if a.submit:
        submit(str(f), name, a.start, a.end, a.baseline)
    if not a.replace:
        return
    _hd, hdf = read_raw(a.h)
    x_eff = xdf
    if a.ewma and a.ewma > 0:
        x_eff = xdf.sort_values(["instrument", "date"]).reset_index(drop=True)
        fac = x_eff["factor"].to_numpy(float)
        a_ = a.ewma
        sm = np.empty(len(x_eff))
        cur, prev = np.nan, None
        for k in range(len(x_eff)):
            if x_eff["instrument"].iat[k] != prev:
                cur, prev = np.nan, x_eff["instrument"].iat[k]
            v = fac[k]
            if np.isfinite(v):
                cur = v if not np.isfinite(cur) else (a_ * cur + (1 - a_) * v)
            sm[k] = cur
        x_eff = x_eff.assign(factor=sm)
    rx = pct_rank(x_eff)
    xs = pd.DataFrame({"date": x_eff["date"].values, "instrument": x_eff["instrument"].values, "rx": rx.values})
    rp = pct_rank(pr)
    ps = pd.DataFrame({"date": pr["date"].values, "instrument": pr["instrument"].values, "rxp": rp.values})
    base = hdf[["date", "instrument", "factor"]].merge(xs, on=["date", "instrument"], how="left")
    base = base.merge(ps, on=["date", "instrument"], how="left")
    base["rx"] = base["rx"].fillna(0.0)
    base["rxp"] = base["rxp"].fillna(0.0)
    v = base["factor"].to_numpy(float) + a.w * (base["rxp"].to_numpy(float) - base["rx"].to_numpy(float))
    v[base["date"].dt.strftime("%H%M").eq("0930").to_numpy()] = 0.0
    o = pd.DataFrame({"date": base["date"].to_numpy(), "instrument": base["instrument"].to_numpy(),
                      "factor": np.nan_to_num(v, nan=0.0)})
    nm = "v_PN%s_raw_w%s" % (a.tag, ("%+.3f" % a.w).replace("+", "p").replace("-", "m").replace(".", ""))
    fo = out / (nm + ".parquet")
    o.to_parquet(fo, index=False)
    print("写出 %s rows=%d" % (fo, len(o)))
    if a.submit:
        submit(str(fo), nm, a.start, a.end, a.baseline)


if __name__ == "__main__":
    main()
