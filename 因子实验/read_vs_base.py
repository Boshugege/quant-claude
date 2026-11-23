# -*- coding: utf-8 -*-
"""按"每一项都不变差"读评测机结果（因子任务或 values 任务都可以）。
用法：python3 因子实验/read_vs_base.py <任务名正则> [--base 基线任务号或任务名，默认 490 = 1012-153145-160b]
输出：每个匹配任务在 2023、2024（或其区间内各年）相对基线的五项差；两年都满足时行首标 ★。
容差（第二轮任务书 §8.1）：IC ≥ −0.0002、IC_IR ≥ −0.005、多空 ≥ −0.3、压力 ≥ −0.005，换手不升。
"""
import argparse, glob, json, os, re

J = ["/home/user/jyc_scorer/jobs", "/srv/media/jyc_scorer/jobs_archive"]
TOL = {"ic_mean": -0.0002, "ic_ir": -0.005, "ls_sharpe": -0.3, "stress": -0.005}
K = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")


def find(spec):
    for b in J:
        if os.path.exists(os.path.join(b, spec, "result.json")):
            return os.path.join(b, spec)
    hits = []
    for b in J:
        for sp in glob.glob(os.path.join(b, "*", "spec.json")):
            try:
                if json.load(open(sp)).get("name") == spec and os.path.exists(sp.replace("spec.json", "result.json")):
                    hits.append(os.path.dirname(sp))
            except Exception:
                pass
    if not hits:
        raise SystemExit("找不到任务 " + spec)
    return sorted(hits, key=os.path.basename)[-1]


ap = argparse.ArgumentParser()
ap.add_argument("pattern")
ap.add_argument("--base", default="1012-153145-160b")
a = ap.parse_args()
base = json.load(open(os.path.join(find(a.base), "result.json")))["by_year"]
pat = re.compile(a.pattern)
rows = []
for b in J:
    for sp in glob.glob(os.path.join(b, "*", "spec.json")):
        try:
            nm = json.load(open(sp)).get("name") or ""
        except Exception:
            continue
        rf = sp.replace("spec.json", "result.json")
        if not pat.search(nm) or not os.path.exists(rf):
            continue
        by = json.load(open(rf)).get("by_year", {})
        parts, ok_all = [], True
        for y in sorted(set(by) & set(base)):
            d = {k: by[y][k] - base[y][k] for k in K}
            ok = all(d[k] >= TOL[k] for k in TOL) and d["turnover"] <= 0
            ok_all &= ok
            parts.append(f"{y}: IC {d['ic_mean']:+.5f} IR {d['ic_ir']:+.4f} LS {d['ls_sharpe']:+.2f} 压力 {d['stress']:+.4f} 换手 {d['turnover']:+.4f}{' ✓' if ok else ''}")
        rows.append((nm, ok_all and bool(parts), " | ".join(parts)))
for nm, ok, s in sorted(set(rows)):
    print(f"{'★' if ok else ' '} {nm:44s} {s}")
