# -*- coding: utf-8 -*-
"""评分机任务的逐年稳定性（10-21：公榜教训——只看 2024 不够）。按任务名前缀列出 2020–2024 每年的 IC／IC_IR／换手，
以及五年里 IC_IR 的最小值、IC 同号的年数、与 90 号（多尺度反转阶梯）的截面秩相关（判断是不是反转的另一种写法）；按最小 IC_IR 排序。用法：python3 c99_years.py s5_ [s4_ ...]"""
import glob
import json
import os
import sys

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
rows = []
for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
    try:
        s = json.load(open(sp, encoding="utf-8"))
    except Exception:
        continue
    n = s.get("name") or ""
    if not any(n.startswith(p) for p in sys.argv[1:]):
        continue
    rp = os.path.join(os.path.dirname(sp), "result.json")
    if not os.path.exists(rp):
        continue
    r = json.load(open(rp, encoding="utf-8"))
    by = r.get("by_year") or {}
    if len(by) < 2:
        continue
    ics = [m["ic_mean"] for m in by.values()]
    sg = 1 if sum(ics) >= 0 else -1
    irs = [sg * m["ic_ir"] for m in by.values()]
    cr = r.get("correlations") or {}
    c90 = (cr.get("90") or {}).get("rank_corr_mean")
    rows.append((min(irs), n, sg, by, c90))
rows.sort(reverse=True)
for mn, n, sg, by, c90 in rows:
    same = sum(1 for m in by.values() if sg * m["ic_mean"] > 0)
    print("%-28s 方向%+d 最小IC_IR %.3f 同号 %d/%d 与90相关 %s | " % (n, sg, mn, same, len(by), "%+.2f" % (sg * c90) if c90 is not None else "—") +
          " ".join("%s %.4f/%.2f/%.2f" % (y[2:], m["ic_mean"], m["ic_ir"], m["turnover"]) for y, m in sorted(by.items())))
