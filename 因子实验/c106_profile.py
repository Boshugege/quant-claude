# -*- coding: utf-8 -*-
"""方向画像（10-21 晚）：1617 的短板是 14:00 时段（2024 年 IC 0.061，其它时段 0.074～0.109）与高离散的日子（2024 年高离散半 IC_IR 1.27，低离散半 1.57）。
找能补短板的方向：读评分机任务的 result.json，打印各时段 IC、高／低离散半的 IC_IR，以及两项"互补度"：
  slot14  = 14:00 的 IC ÷ 各时段 IC 均值（> 1 表示 14:00 相对更强）
  hd      = 高离散半 IC_IR ÷ 低离散半 IC_IR（> 1 表示高离散的日子反而更稳）
用法：python3 c106_profile.py <前缀> [<前缀> ...]（按 hd 排序）"""
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
    d = r.get("detail") or {}
    bs = d.get("by_slot") or {}
    if not bs:
        continue
    ics = {k: v["ic_mean"] for k, v in bs.items()}
    sg = 1.0 if sum(ics.values()) >= 0 else -1.0
    ics = {k: sg * v for k, v in ics.items()}
    mean = sum(ics.values()) / max(len(ics), 1)
    hi, lo = d.get("stress_high_dispersion_ic_ir"), d.get("stress_low_dispersion_ic_ir")
    if hi is None or lo is None or not mean:
        continue
    hi, lo = sg * hi, sg * lo
    slot14 = ics.get("14:00", 0.0) / mean if mean else 0.0
    hd = hi / lo if lo else 0.0
    rows.append((hd, n, slot14, hi, lo, ics, r["metrics"]["turnover"], s.get("start", "")[:4] + "–" + s.get("end", "")[:4]))
rows.sort(reverse=True)
print("%-30s %-9s %6s %6s %6s %6s %5s | %s" % ("任务", "区间", "hd", "高离散", "低离散", "14:00比", "换手", "各时段 IC"))
for hd, n, s14, hi, lo, ics, to, span in rows:
    print("%-30s %-9s %6.2f %6.3f %6.3f %6.2f %5.2f | %s" % (n[:30], span, hd, hi, lo, s14, to, " ".join("%s:%.3f" % (k[:5], v) for k, v in sorted(ics.items()))))
