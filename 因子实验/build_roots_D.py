# -*- coding: utf-8 -*-
"""方向 D（已实现高阶矩）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1118_日内分钟收益偏度.py")
t0 = io.open(BASE, encoding="utf-8").read()

VAR = [
    (1119, "日内分钟收益峰度", [('STAT = "skew"            # skew | kurt | dnratio | skew_m_kurt',
                            'STAT = "kurt"            # skew | kurt | dnratio | skew_m_kurt')]),
    (1120, "日内分钟下行半方差占比", [('STAT = "skew"            # skew | kurt | dnratio | skew_m_kurt',
                                'STAT = "dnratio"         # skew | kurt | dnratio | skew_m_kurt')]),
    (1121, "日内分钟偏度减峰度", [('STAT = "skew"            # skew | kurt | dnratio | skew_m_kurt',
                             'STAT = "skew_m_kurt"     # skew | kurt | dnratio | skew_m_kurt')]),
    (1122, "日内分钟收益偏度五日滚动", [("ROLL_DAYS = 0            # 0 = 只用当日累计；5 = 把当日统计量再取过去 5 日均值",
                                  "ROLL_DAYS = 5            # 0 = 只用当日累计；5 = 把当日统计量再取过去 5 日均值")]),
    (1123, "日内分钟下行半方差占比五日滚动", [('STAT = "skew"            # skew | kurt | dnratio | skew_m_kurt',
                                    'STAT = "dnratio"         # skew | kurt | dnratio | skew_m_kurt'),
                                   ("ROLL_DAYS = 0            # 0 = 只用当日累计；5 = 把当日统计量再取过去 5 日均值",
                                    "ROLL_DAYS = 5            # 0 = 只用当日累计；5 = 把当日统计量再取过去 5 日均值")]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向D变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
