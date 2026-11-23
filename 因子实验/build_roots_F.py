# -*- coding: utf-8 -*-
"""方向 F（因果日内价格位置）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1130_因果日内相对位置.py")
t0 = io.open(BASE, encoding="utf-8").read()
VAR = [
    (1131, "对当日累计VWAP偏离", [('STAT = "pos_range"', 'STAT = "dev_vwap"')]),
    (1132, "对当日累计VWAP偏离除日振幅", [('STAT = "pos_range"', 'STAT = "dev_vwap_norm"')]),
    (1133, "距当日最低幅度", [('STAT = "pos_range"', 'STAT = "dist_low"')]),
    (1134, "距当日最高幅度", [('STAT = "pos_range"', 'STAT = "dist_high"')]),
    (1135, "相对位置与VWAP偏离秩合成", [('STAT = "pos_range"', 'STAT = "blend"')]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向F变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
