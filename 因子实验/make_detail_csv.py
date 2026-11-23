# -*- coding: utf-8 -*-
"""把 工具/评分机/结果/*/result.json 里 2023-01-01~2024-12-31 的本地跑分补进
文档/因子跑分明细_2026-10-13.csv（该文件是 工具/因子图谱/build.py 的本地指标来源）。
只**追加**缺失行，不修改任何既有行；family 一列：新编号（>=1060）标 L_精选优化，其余标 Z_其他。
用法：python3 因子实验/make_detail_csv.py [--dry-run]
"""
import csv, glob, json, os, re, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
CSV = os.path.join(ROOT, "文档", "因子跑分明细_2026-10-13.csv")
COLS = ["num", "factor_file", "family", "start", "end", "created", "ic_mean", "ic_ir", "ls_sharpe",
        "stress", "turnover", "corr104_rank", "job"]

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
have = set()
for r in rows:
    have.add((r["job"] or "").strip())
    if (r["factor_file"] or "").strip():
        have.add(r["factor_file"].strip())

add = []
for rj in sorted(glob.glob(os.path.join(ROOT, "工具", "评分机", "结果", "*", "result.json"))):
    j = json.load(open(rj, encoding="utf-8"))
    m = j.get("meta") or {}
    metrics = j.get("metrics") or {}
    if m.get("start") != "2023-01-01" or m.get("end") != "2024-12-31":
        continue
    if not metrics.get("ic_mean"):
        continue
    inputs = m.get("inputs") or []
    ffile = inputs[0] if inputs else ""
    job = m.get("name") or ""
    if job in have or (ffile and ffile in have):
        continue
    mm = re.match(r"(\d+)_", ffile)
    num = mm.group(1) if mm else ""
    fam = "L_精选优化" if (num and int(num) >= 1060) else "Z_其他"
    corr = ((j.get("correlations") or {}).get("104") or {}).get("rank_corr_mean")
    add.append({
        "num": num, "factor_file": ffile, "family": fam, "start": m["start"], "end": m["end"],
        "created": m.get("created") or "", "ic_mean": metrics.get("ic_mean"), "ic_ir": metrics.get("ic_ir"),
        "ls_sharpe": metrics.get("ls_sharpe"), "stress": metrics.get("stress"), "turnover": metrics.get("turnover"),
        "corr104_rank": "" if corr is None else corr, "job": job,
    })

print("新增 %d 行（既有 %d 行）" % (len(add), len(rows)))
for a in add:
    print("  +", a["num"] or "-", a["job"], a["factor_file"])
if "--dry-run" in sys.argv:
    raise SystemExit(0)
with open(CSV, "a", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLS)
    for a in add:
        w.writerow(a)
print("已追加到", CSV)
