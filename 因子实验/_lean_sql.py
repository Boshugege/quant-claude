# -*- coding: utf-8 -*-
"""分钟 SQL 裁剪：只保留所选 STAT 用到的 m3 / m2 列，sig 字典改成按需计算。build_multiroot.py、build_standalone_monthly.py 共用。"""
import re


def _split_top(t):
    """按顶层逗号切分 SELECT 列表（括号里的逗号不切）。"""
    out, dep, cur = [], 0, ""
    for ch in t:
        if ch == "(":
            dep += 1
        elif ch == ")":
            dep -= 1
        if ch == "," and dep == 0:
            out.append(cur); cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def lean_root(root, stats):
    """只保留 stats 用到的 m3 / m2 列；sig 字典改成按需计算（lambda），算法逐字不变。"""
    sig_lines = dict(re.findall(r'^\s+"(\w+)": (.*),$', root.split("sig = {", 1)[1].split("}[STAT", 1)[0], flags=re.M))
    need = {"ca", "cv", "cabs"}                                             # vw、cabs 在字典之前就会用到
    for st in stats:
        need |= set(re.findall(r'df\["(\w+)"\]', sig_lines[st.split("_res")[0]]))
    i0 = root.index("m3 AS (")
    i1 = root.index("SELECT ", i0) + len("SELECT ")
    i2 = root.index("\n            FROM m2", i1)
    items = _split_top(root[i1:i2])
    base = {"instrument_id", "instrument", "d", "hm", "date", "c"}
    keep = [it for it in items if it.strip() in base or re.search(r"\bAS (\w+)\s*$", it.strip()) and re.search(r"\bAS (\w+)\s*$", it.strip()).group(1) in need]
    got = {it.strip() if it.strip() in base else re.search(r"\bAS (\w+)\s*$", it.strip()).group(1) for it in keep}
    assert need <= got, need - got
    m3 = ",".join(keep)
    root = root[:i1] + m3 + root[i2:]
    # m2：SELECT *, … FROM m1 —— 只保留 m3 里引用到的派生列
    j0 = root.index("m2 AS (")
    j1 = root.index("SELECT *,", j0) + len("SELECT *,")
    j2 = root.index("\n            FROM m1", j1)
    used = set(re.findall(r"\b(\w+)\b", m3))
    its2 = _split_top(root[j1:j2])
    keep2 = [it for it in its2 if re.search(r"\bAS (\w+)\s*$", it.strip()).group(1) in used]
    if not keep2:
        root = root[:j1 - 1] + root[j2:]                                    # 去掉 "SELECT *," 末尾逗号
    else:
        root = root[:j1] + ",".join(keep2) + root[j2:]
    # sig 字典只留所选 STAT 的条目，并按需计算（未选中的条目删掉，自检就不会报"引用了不存在的列"）
    keep_st = {st.split("_res")[0] for st in stats}
    blk0 = root.index("sig = {")
    blk1 = root.index("}[STAT", blk0)
    blk = "\n".join(ln for ln in root[blk0:blk1].split("\n")
                    if not re.match(r'^\s+"(\w+)": ', ln) or re.match(r'^\s+"(\w+)": ', ln).group(1) in keep_st)
    blk = re.sub(r'^(\s+"\w+": )(.*),$', r"\1lambda: \2,", blk, flags=re.M)
    root = root[:blk0] + blk + root[blk1:]
    root, k_ = re.subn(r'\}\[STAT\.split\("_res"\)\[0\]\]', '}[STAT.split("_res")[0]]()', root, count=1)
    assert k_ == 1
    if not any("_res" in st for st in stats) and '        if "_res" in STAT:' in root:   # 没有选残差版：去掉残差分支
        r0 = root.index('        if "_res" in STAT:')
        r1 = root.index('        df = df[["date", "instrument", "signal"]]', r0)
        root = root[:r0] + root[r1:]
    return root
