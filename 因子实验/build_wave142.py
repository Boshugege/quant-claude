# -*- coding: utf-8 -*-
"""第一百四十二批：时点重心（c_utime）轴的延续与交互探索。
- 803：789 的 c_utime 权重提到 0.45。
- 804：766 用 c_utime 直接置换条件反转分量（不加真换手）。
- 805：纯源 c_utime × 条件反转 交互。
- 806：纯源 c_utime × 上影线占比 交互。
- 807：纯源 c_utime + 真换手 + 行业滞后 + 行业同伴 四源合成。
- 808：801（平滑时点重心头部）加 2.25 倍中性化探针。
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
S789 = (D / "789_头部换手角加时点重心030.py").read_text(encoding="utf-8")
S801 = (D / "801_头部换手角加平滑时点重心015.py").read_text(encoding="utf-8")
LADDER = '    result["ladder"] = peer_ladder()\n'
SIG789 = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670", "c_utime"],\n'
          '                                  [1.00, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20, 0.30])\n')
NEU801 = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


def add_keep(s, cols, num):
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
    assert m, num + "_keep"
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


def set_signal(s, cols, weights, num):
    new = '    result["signal"] = rank_blend([%s],\n                                  [%s])\n' % (
        ", ".join('"%s"' % c for c in cols), ", ".join("%.2f" % w for w in weights))
    return repl(s, SIG789, new, num + "_sig")


def add_interaction(s, col, a, b, num):
    code = ('    _ra = (result.groupby("date")["%s"].rank(pct=True) - 0.5)\n'
            '    _rb = (result.groupby("date")["%s"].rank(pct=True) - 0.5)\n'
            '    result["%s"] = _ra * _rb\n') % (a, b, col)
    s = repl(s, LADDER, LADDER + code, num + "_inter")
    return s  # 交互列是 result 列，不能进 keep（否则 df 选列时 KeyError）


# 803 c_utime 0.45
s = repl(S789, SIG789, SIG789.replace("0.30])", "0.45])"), "803_sig")
(D / "803_头部换手角时点重心045.py").write_text(s, encoding="utf-8"); print("803")

# 804 置换分量（c_utime 直接替条件反转）
s = set_signal(S789, ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_utime"],
               [1.0, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20], "804")
(D / "804_头部换手角时点重心置换条件反转无真换手.py").write_text(s, encoding="utf-8"); print("804")

# 805/806/807 纯源
s = add_interaction(S789, "u_x_cond", "c_utime", "c_condrev", "805")
s = set_signal(s, ["u_x_cond"], [1.0], "805")
(D / "805_纯源时点重心乘条件反转两点零.py").write_text(s, encoding="utf-8"); print("805")

s = add_interaction(S789, "u_x_upsh", "c_utime", "c_upsh", "806")
s = set_signal(s, ["u_x_upsh"], [1.0], "806")
(D / "806_纯源时点重心乘上影线占比两点零.py").write_text(s, encoding="utf-8"); print("806")

s = set_signal(S789, ["c_utime", "c_turnself", "c_indrev", "c_indpeer"], [1.0, 1.0, 1.0, 1.0], "807")
(D / "807_纯源时点重心真换手行业滞后行业同伴四源两点零.py").write_text(s, encoding="utf-8"); print("807")

# 808 平滑时点重心头部 2.25 中性化
s = repl(S801, NEU801, '    result["factor"] = neutralize("signal", CHARS4, 2.25)\n', "808_neu")
(D / "808_头部平滑时点重心两点二五中性化探针.py").write_text(s, encoding="utf-8"); print("808")
