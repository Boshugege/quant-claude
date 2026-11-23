# -*- coding: utf-8 -*-
"""附加项筛选（10-21 晚）：1617 + w·z(信号) 中，"最后 10 分钟委托结构"五项全好且换手不升（线性加，不进树）。
系统地把所有五年任务逐个以权重 w 加到底座上（逐截面 z 分数相加后再标准化），交评分机看两年（2023–24）五项。
用法：START=2023-01 END=2024-12 python c107_addon.py <前缀> <底座任务1+底座任务2> <w> <信号任务前缀> [...]"""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
OUTD = "/srv/media/jyc_scorer/incoming/addon"; os.makedirs(OUTD, exist_ok=True)
START, END = os.environ.get("START", "2023-01"), os.environ.get("END", "2024-12")


def jobs(prefixes):
    out = {}
    for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
        try:
            s = json.load(open(sp, encoding="utf-8"))
        except Exception:
            continue
        n, d = s.get("name") or "", os.path.dirname(sp)
        if any(n.startswith(p) for p in prefixes) and s.get("start", "")[:4] == "2020" and os.path.exists(os.path.join(d, "factor_raw.parquet")):
            if n not in out or d > out[n]:
                out[n] = d
    return out


def job_dir(name):
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
    return best


def zcol(df, col):
    g = df.groupby("date")[col]
    return ((df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)


prefix, base, w = sys.argv[1], sys.argv[2].split("+"), float(sys.argv[3])
B = pd.concat([pd.read_parquet(os.path.join(job_dir(n), "factor_raw.parquet"))[["date", "instrument", "factor"]] for n in base], ignore_index=True)
B["date"] = pd.to_datetime(B["date"])
B = B.drop_duplicates(["date", "instrument"], keep="last")
B = B[(B["date"] >= pd.Timestamp(START + "-01")) & (B["date"] <= pd.Timestamp(END + "-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23))].reset_index(drop=True)
B["zb"] = zcol(B, "factor")
z930 = (B["date"].dt.hour * 100 + B["date"].dt.minute) == 930
for n, d in sorted(jobs(sys.argv[4:]).items()):
    X = pd.read_parquet(os.path.join(d, "factor_raw.parquet"))[["date", "instrument", "factor"]]
    X["date"] = pd.to_datetime(X["date"])
    X = X.drop_duplicates(["date", "instrument"], keep="last").rename(columns={"factor": "x"})
    M = B[["date", "instrument", "zb"]].merge(X, on=["date", "instrument"], how="left")
    M["x"] = M["x"].astype(float).fillna(0.0)
    M["zx"] = zcol(M, "x")
    M["t"] = M["zb"] + w * M["zx"]
    M["factor"] = zcol(M, "t")
    M.loc[z930.to_numpy(), "factor"] = 0.0
    name = "%s_%s" % (prefix, n)
    f = os.path.join(OUTD, "%s.parquet" % name)
    M[["date", "instrument", "factor"]].to_parquet(f, index=False)
    r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=%s" % START, "-F", "end=%s" % END,
                        "-F", "name=%s" % name, "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print("提交", name, r.stdout.strip()[:40], flush=True)
