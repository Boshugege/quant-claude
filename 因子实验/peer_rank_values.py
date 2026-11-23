# -*- coding: utf-8 -*-
"""D1 同伴化：把某个根的 factor_raw 转成"行业内百分位排名 − 0.5"，再在底座上做替换/叠加。

行业用暴露表 cpt_jyc_2026_exposure 的 industry_level1_code，**按股票 shift(1)**（只取严格早于
截面日期的最近一条，照抄 1083 的写法）。本地暴露表在 /home/user/jyc_scorer/data/exposure/。

两种用法：
  1) 行业排名版（独立因子，命名 pi_<编号>），作为 values 任务提交：
       peer_rank_values.py --src fm_1232 --tag 1232 --start 2023-01 --end 2024-12 --submit
  2) 替换（不是叠加）：在底座 H 上把根 X 换成它的行业排名版
       peer_rank_values.py --replace --h 1015-203626-f5f0 --src fm_1232 --ewma 0.75 --w 0.20 \
           --pi pi_1232 --tag D1R1232 --start 2023-01 --end 2024-12 --submit
     v = H原始值 + w·(rank(pi) − rank(X_eff))，X_eff 对 L_ 类根先做同一股票连续截面 EWMA(α)。
     （因为 H 里本来就含有 w·rank(X_eff) 这一项，减掉它就等于"换成"。）

必须用 /home/user/jyc_scorer/.venv/bin/python，并用 agent-run 包一层。
"""
import argparse
import glob
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from build_values_composite import find_job, read_raw, pct_rank  # noqa: E402

EXPOSURE = sorted(glob.glob("/home/user/jyc_scorer/data/exposure/exposure_*.parquet"))
URL = "http://127.0.0.1:46220/api/jobs"


def prior_industry():
    """(instrument, day) → 该股票"严格早于该交易日"的最近一条申万一级行业。"""
    parts = []
    for f in EXPOSURE:
        parts.append(pd.read_parquet(f, columns=["date", "instrument", "industry_level1_code"]))
    e = pd.concat(parts, ignore_index=True)
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e["instrument"] = e["instrument"].astype(str)
    e = e.sort_values(["instrument", "day"]).drop_duplicates(["instrument", "day"]).reset_index(drop=True)
    e["ind"] = e.groupby("instrument", sort=False)["industry_level1_code"].shift(1)
    e = e.dropna(subset=["ind"])
    return e.set_index(["instrument", "day"])["ind"]


def industry_rank(df, key):
    """df: date/instrument/factor；按（截面日期，前一交易日行业）做百分位排名 − 0.5，行业缺失记 0。"""
    df = df.copy()
    df["day"] = pd.to_datetime(df["date"]).dt.normalize()
    df["ind"] = key.reindex(pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])).to_numpy()
    g = df.groupby(["date", "ind"], dropna=True)["factor"]
    df["factor"] = (g.rank(pct=True) - 0.5)
    df["factor"] = df["factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # 09:30 与评测机口径一致（评测机本身跳过 09:30 截面，这里显式置 0）
    df.loc[pd.to_datetime(df["date"]).dt.strftime("%H%M") == "0930", "factor"] = 0.0
    return df[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


def ewma_same_stock(xdf, a):
    xdf = xdf.sort_values(["instrument", "date"]).reset_index(drop=True)
    fac = xdf["factor"].to_numpy(float)
    insts = xdf["instrument"].to_numpy()
    sm = np.empty(len(xdf))
    cur, prev = np.nan, None
    for k in range(len(xdf)):
        if insts[k] != prev:
            cur, prev = np.nan, insts[k]
        v = fac[k]
        if np.isfinite(v):
            cur = v if not np.isfinite(cur) else (a * cur + (1 - a) * v)
        sm[k] = cur
    xdf["factor"] = sm
    return xdf


def submit(parquet, name, start, end, baseline, outdir):
    cmd = ["curl", "-s", "-F", "files=@%s" % parquet, "-F", "kind=values",
           "-F", "start=%s" % start, "-F", "end=%s" % end,
           "-F", "name=%s" % name, "-F", "baseline=%s" % baseline, URL]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print("提交 %s -> %s" % (name, r.stdout.strip() or r.stderr.strip()))
    if '"job_id"' in r.stdout:
        pathlib.Path(parquet).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="被同伴化的根（任务名或任务号）")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--outdir", default="/srv/media/jyc_scorer/vc_tmp")
    ap.add_argument("--replace", action="store_true", help="在底座 H 上做替换而不是只做行业排名版")
    ap.add_argument("--h", default="", help="底座任务（--replace 时用）")
    ap.add_argument("--w", type=float, default=0.0, help="该根在底座里的权重（--replace 时用）")
    ap.add_argument("--ewma", type=float, default=0.0, help="L_ 类根在同一股票连续截面上先做 EWMA(α)")
    ap.add_argument("--pi", default="", help="已存在的行业排名版任务（--replace 时用；不给就先自己算）")
    ap.add_argument("--baseline", default="1012-153145-160b")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    out = pathlib.Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    key = prior_industry()

    xd, xdf = read_raw(args.src)
    print("SRC=%s(%s) rows=%d" % (args.src, xd.name, len(xdf)))
    pi = industry_rank(xdf, key)

    if not args.replace:
        name = "pi_%s" % args.tag
        f = out / ("v_%s.parquet" % name)
        pi.to_parquet(f, index=False)
        print("写出 %s rows=%d" % (f, len(pi)))
        if args.submit:
            submit(str(f), name, args.start, args.end, args.baseline, out)
        return

    hd, hdf = read_raw(args.h)
    x_eff = ewma_same_stock(xdf, args.ewma) if args.ewma and args.ewma > 0 else xdf.copy()
    rx = pct_rank(x_eff)
    xs = pd.DataFrame({"date": x_eff["date"].values, "instrument": x_eff["instrument"].values, "rx": rx.values})
    if args.pi:
        _pp = pathlib.Path(args.pi)
        if _pp.suffix == ".parquet" and _pp.exists():          # 允许直接给本地 pi parquet
            pv = pd.read_parquet(_pp)
            pv["date"] = pd.to_datetime(pv["date"])
            pv["instrument"] = pv["instrument"].astype(str)
        else:
            _pid, pv = read_raw(args.pi)
    else:
        pv = pi
    rp = pct_rank(pv)
    ps = pd.DataFrame({"date": pv["date"].values, "instrument": pv["instrument"].values, "rxp": rp.values})
    base = hdf[["date", "instrument", "factor"]].merge(xs, on=["date", "instrument"], how="left")
    base = base.merge(ps, on=["date", "instrument"], how="left")
    base["rx"] = base["rx"].fillna(0.0)
    base["rxp"] = base["rxp"].fillna(0.0)
    v = base["factor"].to_numpy(float) + args.w * (base["rxp"].to_numpy(float) - base["rx"].to_numpy(float))
    v[base["date"].dt.strftime("%H%M").eq("0930").to_numpy()] = 0.0
    o = pd.DataFrame({"date": base["date"].to_numpy(), "instrument": base["instrument"].to_numpy(),
                      "factor": np.nan_to_num(v, nan=0.0)})
    name = "v_%s_raw_w%s" % (args.tag, ("%+.3f" % args.w).replace("+", "p").replace("-", "m").replace(".", ""))
    f = out / (name + ".parquet")
    o.to_parquet(f, index=False)
    print("写出 %s rows=%d" % (f, len(o)))
    if args.submit:
        submit(str(f), name, args.start, args.end, args.baseline, out)


if __name__ == "__main__":
    main()
