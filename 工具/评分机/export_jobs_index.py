# -*- coding: utf-8 -*-
"""10-31 收尾：把评分机全部任务（在线 jobs + 硬盘归档 jobs_archive）导出成一张索引表，作为过程痕迹入库。
每行：任务号、名字、类型、区间、创建时间、输入文件名、五项指标（平台口径），以及分年五项（json 字符串）。
因子值本身（factor_raw.parquet）是可复现的中间产物，不导出。
用法（personal 上）：python3 export_jobs_index.py <输出 csv>"""
import csv
import glob
import json
import os
import sys

ROOTS = [os.path.expanduser("~/jyc_scorer/jobs"), "/srv/media/jyc_scorer/jobs_archive"]
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
out = sys.argv[1]
rows = []
for root in ROOTS:
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        sp = os.path.join(d, "spec.json")
        if not os.path.exists(sp):
            continue
        try:
            s = json.load(open(sp, encoding="utf-8"))
        except Exception:
            continue
        st = {}
        if os.path.exists(os.path.join(d, "state.json")):
            try:
                st = json.load(open(os.path.join(d, "state.json"), encoding="utf-8"))
            except Exception:
                st = {}
        m, by_year = {}, {}
        rp = os.path.join(d, "result.json")
        if os.path.exists(rp):
            try:
                r = json.load(open(rp, encoding="utf-8"))
                m = r.get("metrics") or {}
                by_year = {str(k): {kk: v.get(kk) for kk in MET} for k, v in (r.get("by_year") or {}).items() if isinstance(v, dict)}
            except Exception:
                pass
        rows.append({
            "job_id": os.path.basename(d), "name": s.get("name"), "kind": s.get("kind"), "start": s.get("start"), "end": s.get("end"),
            "label": s.get("label"), "created": st.get("created", ""), "status": st.get("status", ""),
            "inputs": "|".join(s.get("inputs") or []),
            **{k: (round(float(m[k]), 6) if isinstance(m.get(k), (int, float)) else "") for k in MET},
            "by_year": json.dumps(by_year, ensure_ascii=False) if by_year else "",
            "where": "online" if root == ROOTS[0] else "archive",
        })
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("任务", len(rows), "有五项", sum(1 for x in rows if x["ic_mean"] != ""), "->", out)
