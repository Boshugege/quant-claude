# -*- coding: utf-8 -*-
"""新方向的等权合成（10-21）：读评分机任务的 factor_raw.parquet（2020–2024 五年任务），每个信号乘方向后逐截面标准化，
等权平均，09:30 置 0，按五年提交评分机（kind=values）。用来看"低相关新信息"合在一起有多少 IC、和反转阶梯相关多少。
用法（personal，评分机 venv）：
  python c101_compose.py <名字> <任务名[:方向]> [<任务名[:方向]> ...]"""
import glob
import json
import os
import subprocess
import sys
import numpy as np
import pandas as pd

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
OUTD = "/srv/media/jyc_scorer/incoming/comp"; os.makedirs(OUTD, exist_ok=True)


def job_file(name):
    best = None
    for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
        try:
            s = json.load(open(sp, encoding="utf-8"))
        except Exception:
            continue
        d = os.path.dirname(sp)
        if s.get("name") == name and s.get("start", "")[:4] == "2020" and os.path.exists(os.path.join(d, "factor_raw.parquet")):
            if best is None or d > best:
                best = d
    if best is None:
        raise FileNotFoundError(name)
    return os.path.join(best, "factor_raw.parquet")


name, specs = sys.argv[1], sys.argv[2:]
acc, n = None, 0
for sp in specs:
    nm, sg = (sp.split(":") + ["1"])[:2]
    d = pd.read_parquet(job_file(nm))[["date", "instrument", "factor"]]
    d["date"] = pd.to_datetime(d["date"])
    d = d.drop_duplicates(["date", "instrument"], keep="last").set_index(["date", "instrument"])["factor"].astype(float) * float(sg)
    g = d.groupby(level=0)
    z = ((d - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
    acc = z if acc is None else acc.add(z, fill_value=0.0)
    n += 1
    print("读入", nm, sg, len(z), flush=True)
x = (acc / n).rename("factor").reset_index()
x.loc[(x["date"].dt.hour * 100 + x["date"].dt.minute) == 930, "factor"] = 0.0
f = os.path.join(OUTD, "%s.parquet" % name)
x.to_parquet(f, index=False)
r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=2020-01", "-F", "end=2024-12",
                    "-F", "name=%s" % name, "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
print("提交", name, r.stdout.strip()[:80], flush=True)
