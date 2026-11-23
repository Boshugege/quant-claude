# -*- coding: utf-8 -*-
"""新根因子合规检查（机械标识检查，任务书 §1）。

黑名单（出现即判不合格）：
- 反转阶梯：peer_ladder、ladder、r_rv*、r_30、r_day，以及自身过去窗口收益的反转写法
- 头部 K 线分量：c_upsh、c_avgtr、c_body、c_ob
- 泄漏分量：c_utime、c_hitime、c_lotime

白名单/允许：neutralize、低通（ewm 平滑）、rank_blend 等后处理；别人的收益（同伴/龙头）可用。

用法：
    /home/user/jyc_scorer/.venv/bin/python 因子实验/root_check.py <文件> [<文件> ...]
返回码：0=全部通过，2=有文件不合格，1=用法错误。
"""
import os
import re
import sys

BANNED_TOKENS = [
    # 反转阶梯与自身收益反转
    "peer_ladder", "ladder", "r_rv4", "r_rv8", "r_rv16", "r_30", "r_day",
    # 头部 K 线分量
    "c_upsh", "c_avgtr", "c_body", "c_ob",
    # 泄漏分量
    "c_utime", "c_hitime", "c_lotime",
]
# 额外机械规则：(正则, 说明)
BANNED_PATTERNS = [
    (re.compile(r"\br_rv\d+\b"), "r_rv* 多尺度反转分量"),
    (re.compile(r"\br_\d+\b"), "r_<数字> 自身收益反转分量"),
    (re.compile(r"\bladder\w*\b"), "ladder* 反转阶梯"),
]
# 允许出现的例外（变量名里带但语义不是头部分量时，显式列在这里）
ALLOW = {
    "smooth_ladder_free": None,
}


def strip_code(text):
    """去掉注释与三引号/单双引号字符串，返回只含代码的文本。"""
    text = re.sub(r'"""[\s\S]*?"""', " ", text)
    text = re.sub(r"'''[\s\S]*?'''", " ", text)
    out = []
    for ln in text.splitlines():
        ln = re.sub(r"#.*$", "", ln)
        ln = re.sub(r'"[^"]*"', '""', ln)
        ln = re.sub(r"'[^']*'", "''", ln)
        out.append(ln)
    return "\n".join(out)


def check(path):
    raw = open(path, encoding="utf-8", errors="ignore").read()
    # 合成版（build_composites.py 生成）逐字节内联了头部代码，头部本来就用 peer_ladder；
    # 机械检查只应针对**新根**那半边，所以从 `def _root_impl(` 起截。
    off = 0
    i = raw.find("def _root_impl(")
    if i >= 0:
        off = raw[:i].count("\n")
        raw = raw[i:]
    code = strip_code(raw)
    hits = []
    for tok in BANNED_TOKENS:
        for m in re.finditer(r"\b" + re.escape(tok) + r"\b", code):
            line = code[:m.start()].count("\n") + 1 + off
            hits.append((tok, line, "标识黑名单"))
    for pat, why in BANNED_PATTERNS:
        for m in pat.finditer(code):
            tok = m.group(0)
            if tok in BANNED_TOKENS:
                continue
            line = code[:m.start()].count("\n") + 1 + off
            hits.append((tok, line, why))
    # 只保留第 1 次出现的每个标识（行号列表最多 3 个）
    bytok = {}
    for tok, line, why in hits:
        bytok.setdefault((tok, why), []).append(line)
    return bytok


def main(argv):
    if not argv:
        print(__doc__)
        return 1
    bad = 0
    for p in argv:
        if not os.path.isfile(p):
            print("!! 文件不存在：%s" % p)
            bad += 1
            continue
        res = check(p)
        name = os.path.basename(p)
        if not res:
            print("[PASS] %s" % name)
            continue
        bad += 1
        print("[FAIL] %s" % name)
        for (tok, why), lines in sorted(res.items()):
            print("        %-14s %s（行 %s）" % (tok, why, ",".join(str(x) for x in sorted(set(lines))[:3])))
    print("检查 %d 个文件，%d 个不合格" % (len(argv), bad))
    return 2 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
