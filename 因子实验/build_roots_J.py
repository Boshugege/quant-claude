# -*- coding: utf-8 -*-
"""方向 J（竞价信息）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1154_竞价额相对自身前五日.py")
t0 = io.open(BASE, encoding="utf-8").read()
VAR = [
    (1155, "竞价跳空", [('COMPONENT = "amt_ratio"', 'COMPONENT = "gap"')]),
    (1156, "竞价跳空乘异常额", [('COMPONENT = "amt_ratio"', 'COMPONENT = "gap_x_amt"')]),
    (1157, "竞价额占前日全天额", [('COMPONENT = "amt_ratio"', 'COMPONENT = "amt_share"')]),
    (1158, "竞量相对自身前五日", [('COMPONENT = "amt_ratio"', 'COMPONENT = "vol_ratio"')]),
    (1159, "竞价跳空乘前日额占比", [('COMPONENT = "amt_ratio"', 'COMPONENT = "gap_x_share"')]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向J变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
