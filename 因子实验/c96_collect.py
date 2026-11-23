# -*- coding: utf-8 -*-
"""汇总评分机按值评分的结果（personal 上跑）：按名字前缀收集各任务 report.md 里的五项、估公榜分、状态，按估分排序。
用法：python3 c96_collect.py <前缀，如 b1_>  [还可多个前缀]"""
import glob, os, re, sys
rows, pend = [], []
for d in sorted(glob.glob(os.path.expanduser("~/jyc_scorer/jobs/*/"))):
    rp = os.path.join(d, "report.md")
    spec = os.path.join(d, "spec.json")
    nm = None
    if os.path.exists(spec):
        m = re.search(r'"name"\s*:\s*"([^"]+)"', open(spec, encoding="utf-8").read())
        nm = m.group(1) if m else None
    if not nm or not any(nm.startswith(p) for p in sys.argv[1:]):
        continue
    if not os.path.exists(rp):
        pend.append(nm); continue
    s = open(rp, encoding="utf-8").read()
    m = re.search(r"IC ([0-9.]+)，IC_IR ([0-9.]+)，多空夏普 ([0-9.]+)，压力 ([0-9.]+)，换手 ([0-9.]+)", s)
    e = re.search(r"粗略估计公榜分 ([0-9.]+)", s)
    if m:
        rows.append((float(e.group(1)) if e else float("nan"), nm, m.groups()))
rows.sort(key=lambda t: -t[0])
print("%-34s %6s  %s" % ("候选", "估分", "IC／IC_IR／多空／压力／换手"))
for est, nm, g in rows:
    print("%-34s %6.3f  %s" % (nm, est, "/".join(g)))
print("完成 %d，未完成 %d %s" % (len(rows), len(pend), pend[:6]))
