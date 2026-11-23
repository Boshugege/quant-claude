# -*- coding: utf-8 -*-
"""评分机产出直接拿来融合（10-20 夜）：底座（1610 或 c92 定权版）+ w·评分机跑出的新信号（平台脚本任务的 factor_raw.parquet）。
各自逐截面标准化后相加，09:30 置 0，提交评分机（kind=values）。在 personal 上用评分机的 venv 跑：
  PREFIX=c1 BASES=m_1610=<parquet>,k30=<parquet> SIGS=s1_rsj_d,s1_skew_d WS=0.1,0.2 GROUPS=lot=s1_rsj_d+s1_skew_d python c97_mix.py"""
import glob
import json
import os
import subprocess
import numpy as np
import pandas as pd

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
PREFIX = os.environ.get("PREFIX", "c1")
OUTD = "/srv/media/jyc_scorer/incoming/%s" % PREFIX; os.makedirs(OUTD, exist_ok=True)


def job_file(name):
    """按任务名找最近一个完成的任务的因子值。"""
    best = None
    for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
        try:
            s = json.load(open(sp, encoding="utf-8"))
        except Exception:
            continue
        d = os.path.dirname(sp)
        if s.get("name") == name and os.path.exists(os.path.join(d, "factor_raw.parquet")):
            if best is None or d > best:
                best = d
    if best is None:
        raise FileNotFoundError(name)
    return os.path.join(best, "factor_raw.parquet")


def zs(df):
    """逐截面标准化（09:30 置 0）。"""
    x = df["factor"].astype(float)
    g = x.groupby(df["date"])
    z = ((x - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    z[(df["date"].dt.hour * 100 + df["date"].dt.minute) == 930] = 0.0
    return z


def load(path):
    d = pd.read_parquet(path)[["date", "instrument", "factor"]]
    d["date"] = pd.to_datetime(d["date"])
    d = d.drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
    d["z"] = zs(d)
    return d.set_index(["date", "instrument"])["z"]


def submit(name, s):
    f = os.path.join(OUTD, "%s.parquet" % name)
    s.rename("factor").reset_index().to_parquet(f, index=False)
    r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=2024-01", "-F", "end=2024-12",
                        "-F", "name=%s_%s" % (PREFIX, name), "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print("提交", name, r.stdout.strip()[:60], flush=True)


bases = dict(kv.split("=", 1) for kv in os.environ["BASES"].split(","))
sigs = [x for x in os.environ.get("SIGS", "").split(",") if x]
groups = dict(kv.split("=", 1) for kv in os.environ.get("GROUPS", "").split(",") if kv)
ws = [float(x) for x in os.environ.get("WS", "0.1,0.2").split(",")]
S = {n: load(job_file(n)) for n in set(sigs) | {m for g in groups.values() for m in g.split("+")}}
for gname, members in groups.items():
    ms = members.split("+")
    v = sum(S[m] for m in ms) / len(ms)
    S[gname] = v / v.groupby(level=0).transform("std").replace(0, np.nan).fillna(1.0)
    S[gname] = S[gname].fillna(0.0)
for bname, bpath in bases.items():
    B = load(bpath)
    for n in sigs + list(groups):
        x = S[n].reindex(B.index).fillna(0.0)
        for w in ws:
            submit("%s_%s_w%02d" % (bname, n.replace("s1_", "").replace("s2_", ""), round(w * 100)), B + w * x)
print("完成", flush=True)
