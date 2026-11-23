# -*- coding: utf-8 -*-
"""第一百零六批：头部引入按风格五分位分组的条件反转（637 BETA / 638 SIZE）作为新分量。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S589 = (CODE / "589_头部加平滑同伴残差015.py").read_text(encoding="utf-8")
COND_OLD = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
assert COND_OLD in S589, "condrev block not found"
BLEND_OLD = 'rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])'
assert BLEND_OLD in S589, "blend not found"
BLEND_NEW = 'rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_condrev"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15])'
for fname, col, desc in [
    ("637_头部加BETA条件反转015.py", "s_BETA", "合法头部 + 按前一日 BETA 五分位分组的条件反转 0.15"),
    ("638_头部加SIZE条件反转015.py", "s_SIZE", "合法头部 + 按前一日 SIZE 五分位分组的条件反转 0.15"),
]:
    COND = f'''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _qv = df["{col}"]
    _q66 = ((_qv.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))'''
    t = S589.replace(COND_OLD, COND, 1).replace(BLEND_OLD, BLEND_NEW, 1)
    lines = t.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip().startswith('"""') and ln.strip() != '"""':
            lines[i] = '    """' + desc + '（暴露表错一天）。'
            break
    (CODE / fname).write_text("\n".join(lines), encoding="utf-8"); print("写出", fname)
