# -*- coding: utf-8 -*-
"""五年融合（10-21）：底座 = 若干评分机任务的因子值按时间拼接（如 1610 代码的 2020–23 与 2024 两个任务），
逐截面标准化后 + Σ w·z(信号)，信号来自五年任务（新方向合成 m_* 或 s*）。按 2020–2024 五年提交评分机，看逐年五项。
环境变量 BASE_SM：底座平滑（压换手，把换手预算让给快信号）。
用法：python c102_mix5y.py <前缀> <底座任务1+底座任务2> <信号任务> <w1,w2,...> [<信号任务> <ws> ...]"""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
OUTD = "/srv/media/jyc_scorer/incoming/mix5y"; os.makedirs(OUTD, exist_ok=True)
START, END = os.environ.get("START", "2020-01"), os.environ.get("END", "2024-12")     # 底座只有部分年份时（如 1617 的 2023–24）改评估区间


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
    if best is None:
        raise FileNotFoundError(name)
    return best


def zload(names):
    parts = []
    for nm in names:
        d = pd.read_parquet(os.path.join(job_dir(nm), "factor_raw.parquet"))[["date", "instrument", "factor"]]
        d["date"] = pd.to_datetime(d["date"])
        parts.append(d)
    d = pd.concat(parts, ignore_index=True).drop_duplicates(["date", "instrument"], keep="last").set_index(["date", "instrument"])["factor"].astype(float)
    g = d.groupby(level=0)
    return ((d - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)


prefix, base = sys.argv[1], sys.argv[2].split("+")
B = zload(base)
BASE_SM = float(os.environ.get("BASE_SM", "0"))                     # >0：底座先按股票跨截面 EWMA（过去的权重），压换手后再加快信号
if BASE_SM > 0:
    b = B.reset_index().sort_values(["instrument", "date"]).reset_index(drop=True)      # 10-21：pandas 分组 ewm 跑了近 6 小时，改成一次线性扫描
    v, ins = b["factor"].to_numpy(float), b["instrument"].to_numpy()
    out = np.empty_like(v)
    a, prev, last = BASE_SM, 0.0, None
    for i in range(len(v)):
        prev = v[i] if ins[i] != last else a * prev + (1 - a) * v[i]
        out[i], last = prev, ins[i]
    b["factor"] = out
    B = b.set_index(["date", "instrument"])["factor"].reindex(B.index)
    g = B.groupby(level=0)
    B = ((B - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
print("底座", base, len(B), "平滑", BASE_SM, flush=True)
args = sys.argv[3:]
for i in range(0, len(args), 2):
    sig, ws = args[i], [float(w) for w in args[i + 1].split(",")]
    X = zload([sig]).reindex(B.index).fillna(0.0)
    for w in ws:
        name = "%s_%s_w%02d" % (prefix, sig, round(w * 100))
        x = (B + w * X).rename("factor").reset_index()
        x.loc[(x["date"].dt.hour * 100 + x["date"].dt.minute) == 930, "factor"] = 0.0
        f = os.path.join(OUTD, "%s.parquet" % name)
        x.to_parquet(f, index=False)
        r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=%s" % START, "-F", "end=%s" % END,
                            "-F", "name=%s" % name, "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
        print("提交", name, r.stdout.strip()[:60], flush=True)
# 底座本身（同样的标准化）作对照
x = B.rename("factor").reset_index()
x.loc[(x["date"].dt.hour * 100 + x["date"].dt.minute) == 930, "factor"] = 0.0
f = os.path.join(OUTD, "%s_base.parquet" % prefix)
x.to_parquet(f, index=False)
r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=%s" % START, "-F", "end=%s" % END,
                    "-F", "name=%s_base" % prefix, "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
print("提交", "%s_base" % prefix, r.stdout.strip()[:60], flush=True)
