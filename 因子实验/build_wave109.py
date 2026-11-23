# -*- coding: utf-8 -*-
"""第一百零九批：同伴校正与条件反转的合成（644 独立源）+ 头部同时引入两个新源（645）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)

# 644：独立新源 = 校正后的同伴残差（取负）+ 风格条件反转（正号）的排名合成
S623 = rd("623_单独按BETA五分位分组的条件反转.py")
OLD = '    result["signal"] = result["c_condrev"]'
assert OLD in S623
NEW = ('    _rp644 = (result.groupby("date")["c_indpeer"].rank(pct=True) - 0.5)\n'
       '    _rc644 = (result.groupby("date")["c_condrev"].rank(pct=True) - 0.5)\n'
       '    result["signal"] = -_rp644 + _rc644   # c_indpeer 实测为负号，取负后与正号的条件反转合成')
t = S623.replace(OLD, NEW, 1)
t = t.replace("（第 t 个截面上把多尺度反转排名在 BETA 五分位内去均值；暴露表错一天）",
              " 与校正后的同伴残差按排名合成（暴露表错一天）", 1)
wr("644_单独同伴校正与条件反转合成.py", t)

# 645：头部 + span10 平滑同伴 0.30 + 条件反转(BETA) 0.15
S589 = rd("589_头部加平滑同伴残差015.py")
COND_OLD = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
COND_NEW = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _qv645 = df["s_BETA"]
    _q66 = ((_qv645.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))'''
BLEND_OLD = 'rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])'
BLEND_NEW = 'rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_condrev"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15])'
assert COND_OLD in S589 and BLEND_OLD in S589
t = S589.replace(COND_OLD, COND_NEW, 1).replace(BLEND_OLD, BLEND_NEW, 1)
lines = t.split("\n")
for i, ln in enumerate(lines):
    if ln.strip().startswith('"""') and ln.strip() != '"""':
        lines[i] = '    """合法头部 + span10 平滑同伴 0.30 + 按 BETA 五分位分组的条件反转 0.15（两个新源同时叠加；暴露表错一天）。'
        break
wr("645_头部加平滑同伴与条件反转合成.py", "\n".join(lines))
