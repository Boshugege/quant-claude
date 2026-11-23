# -*- coding: utf-8 -*-
"""第五十五批（382+）：把 258 与 353 两条信号做**截面排名线性混合**（集成），以及中性化强度扫描。

动机：353 相对 258 换手大降（z=−49）、IC 升，但 IC_IR 显著变差（z=−2.3）；258 的 IC_IR 更高。
平台头部靠 IC_IR/压力/换手区分，因此寻找"IC_IR 不掉且换手低"的中间点。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "353_同伴上影线平均单笔二日尺度与实体占比.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
    return t.replace(old, new, 1)


SIG353 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
NEU15 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'


def blend(wb, num):
    new = ('    result["sig_a"] = rank_blend(["ladder", "c_upsh", "c_avgtr"], [1.0, 0.30, 0.15])\n'
           '    result["sig_b"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
           '    _ra = result.groupby("date")["sig_a"].rank(pct=True) - 0.5\n'
           '    _rb = result.groupby("date")["sig_b"].rank(pct=True) - 0.5\n'
           '    result["signal"] = %.2f * _ra + %.2f * _rb\n' % (1.0 - wb, wb))
    return repl(BASE.read_text(encoding="utf-8"), SIG353, new, num + "_blend")


def neu(frac, num):
    return repl(BASE.read_text(encoding="utf-8"), NEU15, '    result["factor"] = neutralize("signal", CHARS4, %.1f)\n' % frac, num + "_neu")


def weight(comp, w, num):
    cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"]
    ws = [1.0, 0.30, 0.15, 0.20, 0.15]
    ws[cols.index(comp)] = w
    sig = '    result["signal"] = rank_blend([%s], [%s])\n' % (", ".join('"%s"' % c for c in cols), ", ".join("%.2f" % x for x in ws))
    return repl(BASE.read_text(encoding="utf-8"), SIG353, sig, num + "_w")


def c388(s):
    """上影线 + 下影线同时进入信号（上行冲高回落 + 下行承接）。"""
    code = ('    df["c_dnsh"] = -roll(df["sum_dnsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')
    s = repl(s, '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n',
             '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n' + code, "388_code")
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
    body = m.group(1).rstrip() + ', "c_dnsh"'
    s = s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_dnsh"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n'
    return repl(s, SIG353, sig, "388_sig")


CANDS = [("382", "信号混合35", lambda: blend(0.35, "382")),
         ("383", "信号混合50", lambda: blend(0.50, "383")),
         ("384", "信号混合65", lambda: blend(0.65, "384")),
         ("385", "中性化强度20", lambda: neu(2.0, "385")),
         ("386", "中性化强度10", lambda: neu(1.0, "386")),
         ("387", "上影线权重022", lambda: weight("c_upsh", 0.22, "387")),
         ("388", "加上下影线", lambda: c388(BASE.read_text(encoding="utf-8")))]

for num, name, fn in CANDS:
    s = fn()
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
