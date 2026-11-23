# -*- coding: utf-8 -*-
"""方向 H（主动买卖）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1142_带号成交额占比.py")
t0 = io.open(BASE, encoding="utf-8").read()
VAR = [
    (1143, "当日累计带号成交额占比", [('WINDOW = "seg"', 'WINDOW = "day"')]),
    (1144, "大单分钟带号净额占比", [('STAT = "net"', 'STAT = "big_net"')]),
    (1145, "当日累计大单分钟带号净额占比", [('STAT = "net"', 'STAT = "big_net"'), ('WINDOW = "seg"', 'WINDOW = "day"')]),
    (1146, "带号成交额占比五日滚动", [("ROLL_DAYS = 0", "ROLL_DAYS = 5")]),
    (1147, "大单分钟带号净额占比五日滚动", [('STAT = "net"', 'STAT = "big_net"'), ("ROLL_DAYS = 0", "ROLL_DAYS = 5")]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向H变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
