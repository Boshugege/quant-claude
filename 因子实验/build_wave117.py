# -*- coding: utf-8 -*-
"""第一百一十七批：头部 span5@0.35+条件0.15（668）+ 多风格条件反转合成源（669）+ 头部引入 669（670）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
# 668：基于 657（span5平滑同伴0.30+条件平滑0.15），同伴权重 0.30→0.35，条件改非平滑
S657 = rd("657_头部加平滑同伴与条件反转平滑合成.py")
t = S657.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.35, 0.15]", 1)
t = t.replace("span5 平滑同伴 0.30 + BETA 条件反转平滑版 0.15", "span5 平滑同伴 0.35 + BETA 条件反转平滑版 0.15", 1)
wr("668_头部加平滑同伴035与条件反转平滑015.py", t)
# 669：多风格条件反转合成（五分组平滑的均值）
S623 = rd("623_单独按BETA五分位分组的条件反转.py")
COND_OLD = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _qv = df["s_BETA"]
    _q66 = ((_qv.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))'''
COND_NEW = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _parts669 = []
    for _col669 in ["s_BETA", "s_SIZE", "s_LIQUIDTY", "s_RESVOL", "s_MOMENTUM"]:
        _qv669 = df[_col669]
        _q669 = ((_qv669.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
        _parts669.append(_rk66 - _rk66.groupby([df["date"], _q669]).transform("mean"))
    df["c_condrev"] = pd.concat(_parts669, axis=1).mean(axis=1)'''
assert COND_OLD in S623
b = S623.replace(COND_OLD, COND_NEW, 1).replace(
    "（第 t 个截面上把多尺度反转排名在 BETA 五分位内去均值；暴露表错一天）",
    "（第 t 个截面上把多尺度反转排名在 BETA/SIZE/LIQUIDTY/RESVOL/MOMENTUM 五个五分位内分别去均值后取平均；暴露表错一天）", 1)
OLD = '    result["signal"] = result["c_condrev"]'
NEW = ('    _ord669 = result.sort_values(["instrument", "date"])\n'
       '    result["signal"] = _ord669.groupby("instrument", sort=False)["c_condrev"].transform(\n'
       '        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)')
b = b.replace(OLD, NEW, 1)
wr("669_单独多风格条件反转合成.py", b)
# 670：头部 + 669（多风格条件反转平滑）0.15
S589 = rd("589_头部加平滑同伴残差015.py")
rb_old = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
rb_new = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _parts670 = []
    for _col670 in ["s_BETA", "s_SIZE", "s_LIQUIDTY", "s_RESVOL", "s_MOMENTUM"]:
        _qv670 = df[_col670]
        _q670 = ((_qv670.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
        _parts670.append(_rk66 - _rk66.groupby([df["date"], _q670]).transform("mean"))
    df["c_condrev"] = pd.concat(_parts670, axis=1).mean(axis=1)'''
blend_old = 'rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])'
blend_new = '''    _ord670 = result.sort_values(["instrument", "date"])
    result["c_conds670"] = _ord670.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15])'''
assert rb_old in S589 and blend_old in S589
t = S589.replace(rb_old, rb_new, 1).replace(blend_old, blend_new, 1)
lines = t.split("\n")
for i, ln in enumerate(lines):
    if ln.strip().startswith('"""') and ln.strip() != '"""':
        lines[i] = '    """合法头部 + span10 平滑同伴 0.30 + 多风格条件反转平滑 0.15（暴露表错一天）。'
        break
wr("670_头部加平滑同伴与多风格条件反转合成.py", "\n".join(lines))
