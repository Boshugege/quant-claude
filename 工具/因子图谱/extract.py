# -*- coding: utf-8 -*-
"""因子图谱数据抽取：读 因子代码/ 全部因子，抽配方、分量、中性化、数据字段、未来信息标记，推断父代（代码行集合 Jaccard 最近的更早因子）与差异。
输出 工具/因子图谱/raw_factors.json（中间文件）。"""
import difflib, json, os, re
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CODE = os.path.join(ROOT, "因子代码")

def read_code(path):
    s = open(path, encoding="utf-8", errors="ignore").read()
    if path.endswith(".ipynb"):
        try:
            nb = json.loads(s)
            s = "\n".join("".join(c["source"]) for c in nb["cells"] if c.get("cell_type") == "code")
        except Exception:
            pass
    return s

files = {}
for f in os.listdir(CODE):
    m = re.match(r"(\d+)_(.+)\.(py|ipynb)$", f)
    if not m:
        continue
    n = int(m.group(1))
    if n not in files or (f.endswith(".py") and files[n].endswith(".ipynb")):
        files[n] = f

pat_u = re.compile(r'transform\("sum"\)\s*/\s*_g\["sum_upsh"\]|_g\["sum_upsh_t"\]\.transform\("sum"\)')
pat_piv = re.compile(r'pivot_table\(index=\["instrument",\s*"day"\]')
pat_daysum = re.compile(r'groupby\(\[\s*"instrument",\s*"day"\s*\][^\n]*\["(sum_amt|sum_vol|_amv|amt)"\]\.transform\(\s*"sum"|_amv\.groupby\(\[df\["instrument"\], df\["day"\]\][^\n]*transform\("sum"\)')

def norm_lines(code):
    out = []
    for ln in code.splitlines():
        t = ln.split("#")[0].strip() if not ln.strip().startswith(("\"", "'")) else ln.strip()
        if t:
            out.append(t)
    return out

recs = {}
for n, f in sorted(files.items()):
    code = read_code(os.path.join(CODE, f))
    name = re.match(r"\d+_(.+)\.(py|ipynb)$", f).group(1)
    lines = norm_lines(code)
    # 最终配方：最后一个 result["signal"] / result["factor"] 赋值附近
    sig = [ln for ln in lines if re.match(r'result\[\s*"(signal|factor)"\s*\]\s*=', ln)]
    blends = re.findall(r'rank_blend\(\s*\[([^\]]*)\]\s*(?:,\s*\[([^\]]*)\])?\s*\)', "\n".join(sig))
    comps = []
    if blends:
        cols, ws = blends[-1]
        cols = [c.strip().strip('"\'') for c in cols.split(",") if c.strip()]
        ws = [float(x) for x in re.findall(r"-?[\d.]+", ws)] if ws else [1.0] * len(cols)
        comps = [[c, ws[i] if i < len(ws) else None] for i, c in enumerate(cols)]
    neu = re.findall(r'neutralize\(\s*"signal"\s*,\s*(\w+)\s*,\s*([\d.]+)', "\n".join(sig))
    defs = sorted(set(re.findall(r'df\["((?:c|r|z|p|s|x)_[A-Za-z0-9_]+)"\]\s*=', code)))
    sqlcols = sorted(set(re.findall(r"\bAS\s+([a-z_][a-z0-9_]*)", code)))
    raw_fields = sorted(set(re.findall(r"\b(bid_volume\d|ask_volume\d|bid_price\d|ask_price\d|bid_num_orders\d|ask_num_orders\d|deal_number|float_market_cap|cpt_jyc_2026_exposure|cpt_jyc_2026_\w+)\b", code)))
    leak = []
    if pat_u.search(code):
        leak.append("c_utime 整天聚合")
    if pat_piv.search(code) and ("nanargmax" in code or "nanargmin" in code):
        leak.append("最高/最低价时段 整天聚合")
    if pat_daysum.search(code):
        leak.append("全天成交额分母（待确认）")
    recs[n] = dict(num=n, file=f, name=name, n_lines=len(lines), signal=sig[-3:], comps=comps,
                   neutral=[[a, float(b)] for a, b in neu], defs=defs, sql=sqlcols, fields=raw_fields, leak=leak,
                   _lines=lines)

# 父代推断：更早因子里代码行集合 Jaccard 最大者
nums = sorted(recs)
sets = {n: set(recs[n]["_lines"]) for n in nums}
for i, n in enumerate(nums):
    best, bs = None, 0.0
    a = sets[n]
    for m in nums[:i]:
        b = sets[m]
        inter = len(a & b)
        if not inter:
            continue
        j = inter / len(a | b)
        if j > bs or (j == bs and m > (best or -1)):
            best, bs = m, j
    r = recs[n]
    r["parent"], r["sim"] = best, round(bs, 3)
    if best is not None:
        pa = recs[best]
        d = list(difflib.unified_diff(pa["_lines"], r["_lines"], lineterm="", n=0))
        d = [x for x in d if x.startswith(("+", "-")) and not x.startswith(("+++", "---"))]
        r["diff_n"] = len(d)
        r["diff"] = [x[:220] for x in d[:40]]
        pc, cc = dict((c, w) for c, w in pa["comps"]), dict((c, w) for c, w in r["comps"])
        ch = []
        for c in cc:
            if c not in pc:
                ch.append(f"+{c}({cc[c]:g})" if cc[c] is not None else f"+{c}")
            elif pc[c] != cc[c] and cc[c] is not None and pc[c] is not None:
                ch.append(f"{c} {pc[c]:g}→{cc[c]:g}")
        for c in pc:
            if c not in cc:
                ch.append(f"−{c}")
        pn = pa["neutral"][-1][1] if pa["neutral"] else None
        cn = r["neutral"][-1][1] if r["neutral"] else None
        if pn != cn and cn is not None:
            ch.append(f"中性化 {pn if pn is not None else '无'}→{cn:g}×")
        nd = sorted(set(r["defs"]) - set(pa["defs"]))
        if nd:
            ch.append("新定义 " + ",".join(nd[:6]))
        r["change"] = ch
    else:
        r["diff_n"], r["diff"], r["change"] = 0, [], []
for r in recs.values():
    del r["_lines"]
json.dump([recs[n] for n in nums], open(os.path.join(os.path.dirname(__file__), "raw_factors.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(len(recs), "factors")
