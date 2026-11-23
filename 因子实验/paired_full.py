# -*- coding: utf-8 -*-
"""离线用评测机自己的 paired() 计算两个已完成任务之间的配对检验，含 composite（综合百分位差）。
用法：/home/user/jyc_scorer/.venv/bin/python 因子实验/paired_full.py <候选名> <基线名> [<年> ...]
任务名是评测机 spec.json 里的 name（如 fm_1090／h1090／y20_1090），"ALL" 默认按 2023-01~2024-12 读；
年份参数会只在那一年的截面上配对（如 2023 2024 分别给两年）。系统 python3 没有 pandas，必须用 venv 的 python。"""
import sys, json, glob, importlib.util
from pathlib import Path
sys.path.insert(0, "/home/user/jyc_scorer")
import numpy as np
import pandas as pd
import refs as R
spec = importlib.util.spec_from_file_location("bscoring", "/home/user/jyc_scorer/build/scoring.py")
S = importlib.util.module_from_spec(spec); spec.loader.exec_module(S)

def jobdir(name):
    for base in ["/home/user/jyc_scorer/jobs", "/srv/media/jyc_scorer/jobs_archive"]:
        for p in glob.glob(base + "/*/spec.json"):
            try: sp = json.load(open(p))
            except Exception: continue
            if sp.get("name") != name: continue
            d = Path(p).parent
            if (d / "series.npz").exists(): return d
    return None

def load(d):
    z = np.load(d / "series.npz")
    return {k: z[k] for k in z.files}

def subset(s, y):
    dt = np.array(s["date"]).astype("datetime64[ns]")
    yr = np.array([str(x)[:4] for x in dt])
    m = (yr == y)
    return {k: v[m] for k, v in s.items()}

PART = {"ic_mean": "IC", "ic_ir": "IC_IR", "ls_sharpe": "LS", "stress": "压力", "turnover": "换手"}
cname, bname = sys.argv[1], sys.argv[2]
years = sys.argv[3:] or ["ALL"]
cd, bd = jobdir(cname), jobdir(bname)
if not cd or not bd:
    print("missing", cname if not cd else "", bname if not bd else ""); sys.exit(1)
pool = R.RefPool()
S_ = load(cd); B_ = load(bd)
for y in years:
    if y == "ALL":
        a, b = S_, B_
        lo, hi = pd.Period("2023-01","M").start_time, pd.Period("2024-12","M").end_time
    else:
        a, b = subset(S_, y), subset(B_, y)
        lo, hi = pd.Period("%s-01"%y,"M").start_time, pd.Period("%s-12"%y,"M").end_time
    try:
        rt = pool.metrics(lo, hi).dropna()
    except Exception as e:
        rt = None; print("refpool 失败", e)
    r = S.paired(None, a, b, bname, ref_table=rt)
    if "error" in r:
        print("%-5s %s" % (y, r["error"])); continue
    c = r.get("composite") or {}
    out = " | ".join("%s %+.5f z=%+.1f" % (PART[k], v["diff"], v["z"]) for k, v in r["diff"].items())
    comp = ("综合百分位 %+.3f ± %.3f z=%+.1f（约当公榜 %+.3f）%s" % (
        c.get("diff", float('nan')), c.get("se", float('nan')), c.get("z", float('nan')),
        c.get("approx_score_diff", float('nan')),
        ("；已被顶格：%s" % "、".join(c["capped_metrics"])) if c.get("capped_metrics") else "")) if c else "无 composite"
    print("%-5s n=%d corr=%.4f\n     %s\n     %s" % (y, r["n_days"], r["daily_ic_corr"], out, comp))
