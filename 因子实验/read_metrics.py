# -*- coding: utf-8 -*-
"""读评测机任务的主区间五项（按任务名前缀，逗号分隔）。用法：python3 因子实验/read_metrics.py fm_130,fm_131"""
import json, glob, os, sys
pre = tuple(sys.argv[1].split(","))
for d in sorted(glob.glob("/home/user/jyc_scorer/jobs/*")):
    try:
        nm = json.load(open(d + "/spec.json")).get("name", "")
    except Exception:
        continue
    if not nm.startswith(pre):
        continue
    f = d + "/result.json"
    if not os.path.exists(f):
        print(nm, "pending"); continue
    m = json.load(open(f))["metrics"]
    print(nm, {k: round(m[k], 4) for k in ("ic_mean", "ic_ir", "turnover", "stress")})
