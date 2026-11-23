# -*- coding: utf-8 -*-
"""第一百三十三批（继续扩条件反转的分组维度，全部走已验证的"反转×条件分组"路径）：
    760 = 反转排名在「换手率相对自身历史」五分位内去均值
    761 = 反转排名在「Amihud 非流动性」五分位内去均值
    762 = 反转排名在「最大单窗口收益 MAX」五分位内去均值
    763 = 反转排名在「平均单笔成交额」五分位内去均值
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 760-763：按非风格维度分组的条件反转 =====
    _rkC = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    for _tagC, _colC in [("turnself", df["c_turnself"]), ("illiq", df["c_illiq"]),
                         ("maxr", df["c_max"]), ("avgtr", df["c_avgtr"])]:
        _qC = pd.qcut(pd.Series(_colC, index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
        df["c_cond_" + _tagC] = _rkC - _rkC.groupby([df["date"], _qC]).transform("mean")

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_cond_turnself", "c_cond_illiq", "c_cond_maxr", "c_cond_avgtr"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("760_自身换手历史五分位条件反转", "c_cond_turnself",
     "反转排名在「换手率相对自身 20 日历史」五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
    ("761_非流动性五分位条件反转", "c_cond_illiq",
     "反转排名在 Amihud 非流动性五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
    ("762_MAX五分位条件反转", "c_cond_maxr",
     "反转排名在最大单窗口收益 MAX 五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
    ("763_单笔均额五分位条件反转", "c_cond_avgtr",
     "反转排名在平均单笔成交额五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
]:
    S2 = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)
