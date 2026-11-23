# -*- coding: utf-8 -*-
"""方向 A（OFI 订单流不平衡）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1100_OFI日均一档当日累计除成交额.py")
t0 = io.open(BASE, encoding="utf-8").read()

VAR = [
    (1101, "OFI日均一档除前日市值", [('OFI_DENOM = "amt"', 'OFI_DENOM = "mcap"'),
                                ('OFI_DENOM = "mcap"      # amt', 'OFI_DENOM = "mcap"      # amt')]),
    (1102, "OFI日均三档除成交额", [("OFI_DEPTH = 1          # 1 或 3", "OFI_DEPTH = 3          # 1 或 3")]),
    (1103, "OFI当段一档除成交额", [('OFI_WINDOW = "day"     # day 或 seg', 'OFI_WINDOW = "seg"     # day 或 seg')]),
    (1104, "OFI当段三档除成交额", [("OFI_DEPTH = 1          # 1 或 3", "OFI_DEPTH = 3          # 1 或 3"),
                              ('OFI_WINDOW = "day"     # day 或 seg', 'OFI_WINDOW = "seg"     # day 或 seg')]),
    (1105, "OFI日均一档除成交额加低通", [("OFI_LOWPASS = 0.0      # 0 表示不低通", "OFI_LOWPASS = 0.75     # 0 表示不低通")]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向A变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
