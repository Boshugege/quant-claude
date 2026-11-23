# -*- coding: utf-8 -*-
"""D1 同伴化扫掠：九个根各做一份"行业内排名版"（pi_），并做单根替换 + 全替换。

替换口径：v = 1363原始值 + Σ w_i·(rank(pi_i) − rank(X_i))（X_i 对 L_ 类先 EWMA）。
因为 1363 = 头部 + Σ w_i·rank(X_i)，减掉原根再加行业排名版就等于"换成行业排名版"。

用法：agent-run /home/user/jyc_scorer/.venv/bin/python 因子实验/d1_peer_sweep.py [--submit]
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from peer_rank_values import prior_industry, industry_rank, ewma_same_stock, submit  # noqa: E402
from build_values_composite import read_raw, pct_rank  # noqa: E402

H_DEFAULT = "1015-203626-f5f0"       # 1363 的 values 版（主区间 2023-01～2024-12）
# (fm 编号, 在 1363 里的权重, L_ 类 EWMA α)
ROOTS = [("1232", +0.20, 0.75), ("1236", +0.10, 0.0), ("1237", -0.10, 0.0),
         ("1260", -0.10, 0.0), ("1263", +0.10, 0.0), ("1323", +0.10, 0.0),
         ("1273", -0.10, 0.0), ("1334", -0.15, 0.0), ("1342", -0.10, 0.0)]
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")


def wtxt(w):
    return ("%+.3f" % w).replace("+", "p").replace("-", "m").replace(".", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--h", default=H_DEFAULT, help="底座任务（默认主区间 1363）")
    ap.add_argument("--prefix", default="fm", help="根任务名前缀：fm_ / o21_ / o20_")
    ap.add_argument("--suffix", default="", help="输出与任务名后缀，例如 O21")
    a = ap.parse_args()
    H, BASELINE = a.h, a.h
    OUT.mkdir(parents=True, exist_ok=True)
    key = prior_industry()

    hd, hdf = read_raw(H)
    print("H=%s rows=%d" % (hd.name, len(hdf)))
    total = pd.DataFrame({"date": hdf["date"].values, "instrument": hdf["instrument"].values,
                          "d": np.zeros(len(hdf))})

    for fid, w, e in ROOTS:
        _xd, xdf = read_raw("%s_%s" % (a.prefix, fid))
        pi = industry_rank(xdf, key)
        pif = OUT / ("v_pi_%s%s.parquet" % (fid, a.suffix))
        pi.to_parquet(pif, index=False)
        if a.submit:
            submit(str(pif), "pi_%s%s" % (fid, a.suffix), a.start, a.end, BASELINE, OUT)

        xe = ewma_same_stock(xdf, e) if e > 0 else xdf.copy()
        rx = pct_rank(xe)
        xa = pd.DataFrame({"date": xe["date"].values, "instrument": xe["instrument"].values, "rx": rx.values})
        rp = pct_rank(pi)
        pa = pd.DataFrame({"date": pi["date"].values, "instrument": pi["instrument"].values, "rp": rp.values})
        m = hdf[["date", "instrument"]].merge(xa, on=["date", "instrument"], how="left")
        m = m.merge(pa, on=["date", "instrument"], how="left")
        m["rx"] = m["rx"].fillna(0.0)
        m["rp"] = m["rp"].fillna(0.0)
        diff = w * (m["rp"].to_numpy(float) - m["rx"].to_numpy(float))
        total["d"] = total["d"].to_numpy(float) + diff

        v = hdf["factor"].to_numpy(float) + diff
        v[pd.to_datetime(hdf["date"]).dt.strftime("%H%M").eq("0930").to_numpy()] = 0.0
        name = "v_D1R%s%s_raw_w%s" % (fid, a.suffix, wtxt(w))
        o = pd.DataFrame({"date": hdf["date"].values, "instrument": hdf["instrument"].values,
                          "factor": np.nan_to_num(v, nan=0.0)})
        f = OUT / (name + ".parquet")
        o.to_parquet(f, index=False)
        print("写出 %s rows=%d" % (f, len(o)))
        if a.submit:
            submit(str(f), name, a.start, a.end, BASELINE, OUT)

    v = hdf["factor"].to_numpy(float) + total["d"].to_numpy(float)
    v[pd.to_datetime(hdf["date"]).dt.strftime("%H%M").eq("0930").to_numpy()] = 0.0
    o = pd.DataFrame({"date": hdf["date"].values, "instrument": hdf["instrument"].values,
                      "factor": np.nan_to_num(v, nan=0.0)})
    f = OUT / ("v_D1ALL%s_raw.parquet" % a.suffix)
    o.to_parquet(f, index=False)
    print("写出 %s rows=%d" % (f, len(o)))
    if a.submit:
        submit(str(f), "v_D1ALL%s_raw" % a.suffix, a.start, a.end, BASELINE, OUT)


if __name__ == "__main__":
    main()
