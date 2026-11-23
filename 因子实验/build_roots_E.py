# -*- coding: utf-8 -*-
"""方向 E（价格约束）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1124_涨跌停带内位置.py")
t0 = io.open(BASE, encoding="utf-8").read()
VAR = [
    (1125, "距涨停距离", [('STAT = "pos_band"', 'STAT = "dist_up"')]),
    (1126, "距最近涨跌停距离", [('STAT = "pos_band"', 'STAT = "dist_min"')]),
    (1127, "当日触涨停时间占比", [('STAT = "pos_band"', 'STAT = "touch_up"')]),
    (1128, "当日涨停开板次数占比", [('STAT = "pos_band"', 'STAT = "open_up"')]),
    (1129, "距涨停距离乘触板强度", [('STAT = "pos_band"', 'STAT = "dist_x_touch"')]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向E变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
