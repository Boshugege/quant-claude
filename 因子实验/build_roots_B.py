# -*- coding: utf-8 -*-
"""方向 B（日内领先滞后）变体生成：只替换参数块，其余逐字节不变。"""
import io, os
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
BASE = os.path.join(D, "1106_日内领先滞后行业前二十市值领先组.py")
t0 = io.open(BASE, encoding="utf-8").read()

VAR = [
    (1107, "日内领先滞后全市场大盘领先组", [('LEAD_GROUP = "ind_top"   # ind_top | mkt_top | ind_wavg',
                                        'LEAD_GROUP = "mkt_top"   # ind_top | mkt_top | ind_wavg')]),
    (1108, "日内领先滞后行业内市值加权全体", [('LEAD_GROUP = "ind_top"   # ind_top | mkt_top | ind_wavg',
                                        'LEAD_GROUP = "ind_wavg"  # ind_top | mkt_top | ind_wavg')]),
    (1109, "日内领先滞后只取正差距", [("CATCH_ONLY = 0           # 1 = 只取 max(0, 领先组收益 − 自己收益)",
                                "CATCH_ONLY = 1           # 1 = 只取 max(0, 领先组收益 − 自己收益)")]),
    (1110, "日内领先滞后用上两段收益", [("LEAD_LAG = 1             # 1 = 上一段（t-1→t），2 = 上两段",
                                "LEAD_LAG = 2             # 1 = 上一段（t-1→t），2 = 上两段")]),
    (1111, "日内领先滞后行业前十分之一领先组", [("LEAD_TOPN = 0.20         # 领先组取前多少比例（按前一日流通市值）",
                                        "LEAD_TOPN = 0.10         # 领先组取前多少比例（按前一日流通市值）")]),
]
for num, name, subs in VAR:
    t = t0
    for a, b in subs:
        assert a in t, (num, a)
        t = t.replace(a, b, 1)
    lines = t.split("\n")
    lines.insert(2, "    【方向B变体 %d：%s】" % (num, name))
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", num, name)
