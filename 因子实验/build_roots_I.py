# -*- coding: utf-8 -*-
"""方向 I（隔夜与日内分解）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1148_隔夜日内分解前日隔夜.py")
t0 = io.open(BASE, encoding="utf-8").read()
VAR = [
    (1149, "隔夜日内分解前日日内", [('COMPONENT = "ovn"', 'COMPONENT = "intra"')]),
    (1150, "隔夜日内分解前日隔夜减日内", [('COMPONENT = "ovn"', 'COMPONENT = "diff"')]),
    (1151, "隔夜日内分解前日隔夜五日", [('COMPONENT = "ovn"', 'COMPONENT = "ovn5"')]),
    (1152, "隔夜日内分解前日日内五日", [('COMPONENT = "ovn"', 'COMPONENT = "intra5"')]),
    (1153, "隔夜日内分解前日差五日", [('COMPONENT = "ovn"', 'COMPONENT = "diff5"')]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向I变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
