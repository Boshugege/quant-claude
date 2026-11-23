# -*- coding: utf-8 -*-
"""第一百三十八批（收口补点，全部走已验证路径）：
    781 = 反转排名在「行业 × 真实换手」二维格内去均值
    782 = 反转排名在「行业内十分位」内去均值（decile 版本）
    783 = 四尺度反转的加权排名合成（强调短尺度）
    784 = 头部（766 配方）+ 全维条件反转分量权重 0.25
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE
BLOCK = '''    # ===== 新轴 781-783 =====
    _turnH = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"].where(df["mcap"] > 0)
    _qtH = pd.qcut(_turnH.groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    _rkH = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    df["c_cond_indturn"] = _rkH - _rkH.groupby([df["date"], df["ind_code"], _qtH]).transform("mean")
    _qdH = (pd.Series(_rkH, index=df.index).groupby(df["date"]).rank(pct=True) * 10).clip(upper=9.999).fillna(-1).astype(int)
    df["c_cond_dec"] = _rkH - _rkH.groupby([df["date"], _qdH]).transform("mean")
    _rkH2 = pd.concat([df.groupby("date")[c].rank(pct=True) * w for c, w in
                       zip(["r_30", "r_rv4", "r_rv8", "r_day"], [1.6, 1.2, 0.8, 0.4])], axis=1).sum(axis=1) / 4.0 - 0.5
    df["c_rkw"] = _rkH2

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_cond_indturn", "c_cond_dec", "c_rkw"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG = '    result["signal"] = result["c_indpeer"]'
assert SIG in base2
for name, col, doc in [
    ("781_行业换手二维格条件反转", "c_cond_indturn",
     "反转排名在「行业 × 真实换手五分位」二维格内去均值；全部暴露表列错一天）"),
    ("782_十分位条件反转", "c_cond_dec",
     "反转排名在全市场十分位内去均值（decile 版本）；全部暴露表列错一天）"),
    ("783_四尺度加权反转", "c_rkw",
     "四尺度反转的加权排名合成（权重 1.6/1.2/0.8/0.4，强调短尺度）；全部暴露表列错一天）"),
]:
    S2 = base2.replace(SIG, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)

S766 = (CODE / "766_头部换手角二点零中性化.py").read_text(encoding="utf-8")
OLD_B = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])'''
assert OLD_B in S766
S784 = S766.replace(OLD_B, '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.25])''', 1)
(CODE / "784_头部换手角源025二点零中性化.py").write_text(S784, encoding="utf-8"); print("写出 784")
