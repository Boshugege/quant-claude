# -*- coding: utf-8 -*-
"""第一百三十四批：
    766 = 703 配方（span5同伴040+条件反转020，换手角）+ 2.0 倍中性化
    767 = 757 但全维条件反转分量不做平滑（原始）
    768 = 反转排名在「低价」五分位内去均值（新分组维度）
    769 = 反转排名在「40 窗口成交额」五分位内去均值（新分组维度）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

S703 = (CODE / "703_头部加span5同伴040与多风格条件反转020.py").read_text(encoding="utf-8")
OLD_N = '    result["factor"] = neutralize("signal", CHARS4, 1.5)'
assert OLD_N in S703
S766 = S703.replace(OLD_N, '    result["factor"] = neutralize("signal", CHARS4, 2.0)', 1)
S766 = S766.replace("合法头部 + span5", "合法头部 + span5", 1)
(CODE / "766_头部换手角二点零中性化.py").write_text(S766, encoding="utf-8"); print("写出 766")

S757 = (CODE / "757_头部全维条件反转.py").read_text(encoding="utf-8")
OLD_SM = '''    _ord670 = result.sort_values(["instrument", "date"])
    result["c_conds670"] = _ord670.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)'''
assert OLD_SM in S757
S767 = S757.replace(OLD_SM, '    result["c_conds670"] = result["c_condrev"]', 1)
S767 = S767.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "头部全维条件反转且该分量不做平滑（原始全维条件反转，探索 IC 上限）；暴露表错一天。", 1)
(CODE / "767_头部全维条件反转不平滑.py").write_text(S767, encoding="utf-8"); print("写出 767")

BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE
BLOCK = '''    # ===== 新轴 768/769：低价 / 成交额分组条件反转 =====
    _rkD = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _sizeD = roll(df["sum_amt"], 40, "sum", 10)
    for _tagD, _colD in [("lowpx", df["c_lowpx"]), ("sizeD", _sizeD)]:
        _qD = pd.qcut(pd.Series(_colD, index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
        df["c_cond_" + _tagD] = _rkD - _rkD.groupby([df["date"], _qD]).transform("mean")

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_cond_lowpx", "c_cond_sizeD"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2
for name, col, doc in [
    ("768_低价五分位条件反转", "c_cond_lowpx",
     "反转排名在低价（-log 价格）五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
    ("769_成交额五分位条件反转", "c_cond_sizeD",
     "反转排名在 40 窗口成交额五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
]:
    S2 = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)
