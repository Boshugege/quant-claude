# -*- coding: utf-8 -*-
"""第一百四十五批：多空夏普臂（481/482 是多空榜首族）抬 IC_IR/压力，同时给头部+时点重心做 2.5 探针。
- 823：481（头部+隔夜跳空，LS 46.0）中性化 1.5 -> 2.0
- 824：482（头部+市值加权行业滞后，LS 46.1）中性化 1.5 -> 2.0
- 825：481 + 2.5 探针
- 826：482 + 2.5 探针
- 827：789（头部+时点重心）2.5 探针
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
A15 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
A20 = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, t.count(old))
    return t.replace(old, new, 1)


def emit(num, name, s):
    p = D / ("%s_%s.py" % (num, name)); p.write_text(s, encoding="utf-8"); print("写出", p.name)


s481 = (D / "481_头部加隔夜跳空.py").read_text(encoding="utf-8")
s482 = (D / "482_头部加市值加权行业滞后.py").read_text(encoding="utf-8")
s789 = (D / "789_头部换手角加时点重心030.py").read_text(encoding="utf-8")

emit("823", "头部加隔夜跳空中性化二点零", repl(s481, A15, A20, "823"))
emit("824", "头部加市值加权行业滞后中性化二点零", repl(s482, A15, A20, "824"))
emit("825", "头部加隔夜跳空中性化二点五探针", repl(s481, A15, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "825"))
emit("826", "头部加市值加权行业滞后中性化二点五探针", repl(s482, A15, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "826"))
emit("827", "头部换手角时点重心二点五探针", repl(s789, A20, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "827"))
