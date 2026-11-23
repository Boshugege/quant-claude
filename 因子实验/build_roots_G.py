# -*- coding: utf-8 -*-
"""方向 G（价格冲击与流动性）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1136_分钟Amihud冲击.py")
t0 = io.open(BASE, encoding="utf-8").read()
VAR = [
    (1137, "分钟Kyle斜率", [('STAT = "amihud"', 'STAT = "kyle"')]),
    (1138, "当日累计Amihud", [('WINDOW = "seg"', 'WINDOW = "day"')]),
    (1139, "当日累计Kyle斜率", [('STAT = "amihud"', 'STAT = "kyle"'), ('WINDOW = "seg"', 'WINDOW = "day"')]),
    (1140, "分钟Amihud五日滚动", [("ROLL_DAYS = 0", "ROLL_DAYS = 5")]),
    (1141, "分钟Kyle斜率五日滚动", [('STAT = "amihud"', 'STAT = "kyle"'), ("ROLL_DAYS = 0", "ROLL_DAYS = 5")]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向G变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
