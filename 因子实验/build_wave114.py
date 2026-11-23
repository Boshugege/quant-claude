# -*- coding: utf-8 -*-
"""第一百一十四批：头部引入条件反转平滑源（656 BETA@0.30）+ 平滑同伴与平滑条件反转双源（657）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
S649 = rd("649_头部加SIZE条件反转平滑015.py")
# 656：BETA 版 + 0.30
t = S649.replace('_qv = df["s_SIZE"]', '_qv = df["s_BETA"]', 1)
t = t.replace("按前一日 SIZE 五分位分组的条件反转平滑版 0.15", "按前一日 BETA 五分位分组的条件反转平滑版 0.30", 1)
t = t.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.30]", 1)
wr("656_头部加BETA条件反转平滑030.py", t)
# 657：span5 平滑同伴 0.30 + BETA 条件反转平滑 0.15
S589 = rd("589_头部加平滑同伴残差015.py")
COND_OLD = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
COND_NEW = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _qv657 = df["s_BETA"]
    _q66 = ((_qv657.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))'''
BLEND_OLD = 'rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])'
BLEND_NEW = '''    _ord657 = result.sort_values(["instrument", "date"])
    result["c_conds657"] = _ord657.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds657"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15])'''
assert COND_OLD in S589 and BLEND_OLD in S589
t = S589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1).replace(COND_OLD, COND_NEW, 1).replace(BLEND_OLD, BLEND_NEW, 1)
lines = t.split("\n")
for i, ln in enumerate(lines):
    if ln.strip().startswith('"""') and ln.strip() != '"""':
        lines[i] = '    """合法头部 + span5 平滑同伴 0.30 + BETA 条件反转平滑版 0.15（两源同取低换手形态；暴露表错一天）。'
        break
wr("657_头部加平滑同伴与条件反转平滑合成.py", "\n".join(lines))
