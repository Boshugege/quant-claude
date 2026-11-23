# -*- coding: utf-8 -*-
"""第一百一十一批：行业条件反转平滑源（648 独立取证）+ 头部引入条件反转平滑源（649）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
S561 = rd("561_单独行业内市值加权同伴残差反转.py")
OLD = '    result["signal"] = result["c_indpeer"]'
NEW = ('    _ord648 = result.sort_values(["instrument", "date"])\n'
       '    result["signal"] = _ord648.groupby("instrument", sort=False)["c_condrev"].transform(\n'
       '        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)')
assert OLD in S561
t = S561.replace(OLD, NEW, 1).replace(
    "行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）",
    "按申万一级行业分组的条件反转（行业内去均值）的 10 窗口 EWMA 平滑版（暴露表错一天）", 1)
wr("648_单独按行业分组的条件反转平滑版.py", t)

# 649：头部 + 640（SIZE 条件反转平滑）0.15，复用 638 的头部并把 c_condrev 换成平滑版
S638 = rd("638_头部加SIZE条件反转015.py")
OLD638 = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_condrev"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15])'''
NEW649 = '''    _ord649 = result.sort_values(["instrument", "date"])
    result["c_conds649"] = _ord649.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds649"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15])'''
assert OLD638 in S638
t = S638.replace(OLD638, NEW649, 1).replace("按前一日 SIZE 五分位分组的条件反转 0.15", "按前一日 SIZE 五分位分组的条件反转平滑版 0.15", 1)
wr("649_头部加SIZE条件反转平滑015.py", t)
