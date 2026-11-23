# -*- coding: utf-8 -*-
"""平台口径的增量（10-21，用户："不要定论"）：c103 的残差 IC 用的是原始标签（未剔除风格、行业），而且只看 IC。
这里把每个信号逐截面对底座做最小二乘、取残差（z(信号) − ρ·z(底座)），把残差当作因子值交给评分机，
由评分机按平台口径（风格 + 行业剔除）给出五项。残差的 IC_IR、压力就是"底座之外还剩多少可用信息"。
用法：START=2023-01 END=2024-12 python c104_resid_submit.py <前缀> <底座任务1+底座任务2> <信号任务> [...]"""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
OUTD = "/srv/media/jyc_scorer/incoming/resid"; os.makedirs(OUTD, exist_ok=True)
START, END = os.environ.get("START", "2023-01"), os.environ.get("END", "2024-12")


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
lv = B.index.get_level_values(0)
B = B[(lv >= pd.Timestamp(START + "-01")) & (lv <= pd.Timestamp(END + "-01") + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23))]
dates = B.index.get_level_values(0)
for sig in sys.argv[3:]:
    X = zload([sig]).reindex(B.index).fillna(0.0)
    rho = (X * B).groupby(dates).transform("mean")                          # z 分数下的逐截面回归系数 = 相关
    r = (X - rho * B).rename("factor").reset_index()
    r.loc[(r["date"].dt.hour * 100 + r["date"].dt.minute) == 930, "factor"] = 0.0
    name = "%s_%s" % (prefix, sig)
    f = os.path.join(OUTD, "%s.parquet" % name)
    r.to_parquet(f, index=False)
    out = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=%s" % START, "-F", "end=%s" % END,
                          "-F", "name=%s" % name, "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print("提交", name, "平均相关 %.3f" % float(rho.groupby(dates).first().mean()), out.stdout.strip()[:50], flush=True)
