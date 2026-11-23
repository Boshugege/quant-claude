# -*- coding: utf-8 -*-
"""汇总评分机结果：读取 工具/评分机/结果/*/result.json，输出对比表（含参照池百分位均值）。"""
import glob
import json
import os

rows = []
for d in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "工具", "评分机", "结果", "*"))):
    f = os.path.join(d, "result.json")
    if not os.path.exists(f):
        continue
    j = json.load(open(f))
    m = j.get("metrics") or {}
    if not m:
        rows.append((os.path.basename(d), (j.get("meta") or {}).get("name") or "?", "失败", j.get("error")))
        continue
    q = (j.get("reference_pool") or {}).get("percentile") or {}
    q = {k: (0.0 if v is None else float(v)) for k, v in q.items()}
    m = {k: (float("nan") if v is None else v) for k, v in m.items()}
    proxy = sum(q.get(k, 0) for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")) / 5 if q else float("nan")
    rows.append((os.path.basename(d), (j.get("meta") or {}).get("name") or "?", "%d" % (m.get("n_xsec", 0) // 484),
                 m.get("ic_mean"), m.get("ic_ir"), m.get("ls_sharpe"), m.get("stress"), m.get("turnover"),
                 q.get("ic_mean"), q.get("ic_ir"), q.get("ls_sharpe"), q.get("stress"), q.get("turnover"), proxy))
hdr = "%-20s %-18s %2s %7s %7s %7s %7s %7s | %5s %5s %5s %5s %5s %6s" % (
    "任务号", "名称", "截", "IC", "IC_IR", "多空", "压力", "换手", "IC%", "IR%", "LS%", "压%", "换%", "代理分")
print(hdr)
print("-" * len(hdr))
for r in rows:
    if len(r) == 4:
        print("%-20s %-18s %s %s" % r)
    else:
        try:
            print("%-20s %-18s %2s %7.4f %7.3f %7.2f %7.3f %7.3f | %5.2f %5.2f %5.2f %5.2f %5.2f %6.3f" % r)
        except TypeError:
            print("%-20s %-18s 字段缺失 %s" % (r[0], r[1], r[2:]))
print("\n代理分 = 参照池五项百分位的均值（与公榜口径最接近的粗略代理，越接近 1 越好）")
