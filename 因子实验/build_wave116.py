# -*- coding: utf-8 -*-
"""第一百一十六批：条件反转的分组维度再扩展（665 LIQUIDTY / 666 RESVOL / 667 MOMENTUM 五分位）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
COND_OLD = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
assert COND_OLD in S561
for fname, col, desc in [
    ("665_单独按LIQUIDTY五分位分组的条件反转.py", "s_LIQUIDTY", "LIQUIDTY"),
    ("666_单独按RESVOL五分位分组的条件反转.py", "s_RESVOL", "RESVOL"),
    ("667_单独按MOMENTUM五分位分组的条件反转.py", "s_MOMENTUM", "MOMENTUM"),
]:
    COND = f'''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _qv = df["{col}"]
    _q66 = ((_qv.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))'''
    b = S561.replace(COND_OLD, COND, 1).replace(
        "行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）",
        f"按前一日 {desc} 五分位分组的条件反转（第 t 个截面上把多尺度反转排名在 {desc} 五分位内去均值；暴露表错一天）", 1)
    b = b.replace('    result["signal"] = result["c_indpeer"]', '    result["signal"] = result["c_condrev"]', 1)
    (CODE / fname).write_text(b, encoding="utf-8"); print("写出", fname)
