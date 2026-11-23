# -*- coding: utf-8 -*-
"""方向 C（同时段异常量）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1112_同时段异常量成交额二十日基准.py")
t0 = io.open(BASE, encoding="utf-8").read()

VAR = [
    (1113, "同时段异常量乘收益符号", [("SIGNED = 0              # 0=原始；1=乘符号；2=乘收益",
                                "SIGNED = 1              # 0=原始；1=乘符号；2=乘收益")]),
    (1114, "同时段异常量乘收益", [("SIGNED = 0              # 0=原始；1=乘符号；2=乘收益",
                              "SIGNED = 2              # 0=原始；1=乘符号；2=乘收益")]),
    (1115, "同时段异常量五日基准", [("BASE_DAYS = 20          # 基准窗口：过去多少个交易日",
                                "BASE_DAYS = 5           # 基准窗口：过去多少个交易日")]),
    (1116, "当日累计异常量二十日基准", [('WINDOW = "seg"          # seg | day（day = 当日累计到该段）',
                                  'WINDOW = "day"          # seg | day（day = 当日累计到该段）')]),
    (1117, "同时段异常笔数二十日基准", [('METRIC = "amt"          # amt | deals',
                                  'METRIC = "deals"        # amt | deals')]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向C变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
