# -*- coding: utf-8 -*-
"""汇总评分机结果里的配对比较（判优劣的唯一依据），按综合百分位 z 排序。

用法：python3 因子实验/paired_table.py [名称前缀]
"""
import glob
import json
import os
import sys
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAT = sys.argv[1] if len(sys.argv) > 1 else ""


def rows():
    for d in sorted(glob.glob(os.path.join(ROOT, "工具/评分机/结果/*/"))):
        rj = os.path.join(d, "result.json")
        rp = os.path.join(d, "report.md")
        if not os.path.exists(rj):
            continue
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        nm = "?"
        if os.path.exists(rp):
            t = open(rp, encoding="utf-8", errors="ignore").read()
            m = re.search(r"(?m)^# 评分报告：(\S+)", t)
            if m:
                nm = m.group(1)
        if PAT and PAT not in nm:
            continue
        p = r.get("paired") or {}
        c = p.get("composite") or {}
        diff = p.get("diff") or {}
        co = r.get("correlations") or {}
        c104 = co.get("104")
        c104 = c104.get("rank_corr_mean") if isinstance(c104, dict) else c104
        yield dict(id=os.path.basename(d.rstrip("/")), name=nm, metrics=r.get("metrics") or {},
                   comp=c, diff=diff, corr104=c104, rp=r.get("reference_pool") or {})


rs = list(rows())
for r in rs:
    r["_z"] = (r["comp"] or {}).get("z")
rs.sort(key=lambda r: (r["_z"] is None, -(r["_z"] or -99)))
print(f"{'name':<26}{'comp差±se':<18}{'z':<8}{'IC':<7}{'IR':<7}{'LS':<7}{'str':<7}{'turn':<7}{'c104':<7}{'任务号'}")
for r in rs:
    m = r["metrics"]
    c = r["comp"] or {}
    if c:
        comp = "%.3f±%.3f" % (c.get("diff", float("nan")), c.get("se", float("nan")))
        z = "%.2f" % (c.get("z") or 0.0)
    else:
        comp, z = "-", "-"
    f = lambda v, p=4: "-" if v is None else ("%.*f" % (p, v))
    print(f"{r['name'][:25]:<26}{comp:<18}{z:<8}{f(m.get('ic_mean')):<7}{f(m.get('ic_ir'),3):<7}"
          f"{f(m.get('ls_sharpe'),1):<7}{f(m.get('stress'),3):<7}{f(m.get('turnover'),3):<7}"
          f"{f(r['corr104'],3):<7}{r['id']}")
print("共 %d 个有结果的候选" % len(rs))
