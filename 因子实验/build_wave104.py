# -*- coding: utf-8 -*-
"""第一百零四批：行业内龙头领先滞后（622）+ 按 BETA/SIZE 五分位分组的条件反转（623/624，不同机制轴）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)

S561 = rd("561_单独行业内市值加权同伴残差反转.py")
OLD = '''    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
OLD_DOC = "行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）"
assert OLD in S561 and OLD_DOC in S561

# ---- 622：行业内龙头领先滞后（同伴权重 = 行业内前一日市值前 20%） ----
LEAD = '''    # ===== 新轴 622：行业内龙头的滞后收益（同伴权重 = 行业内前一日流通市值前 20%，等权） =====
    _lw622 = df["mcap"].fillna(0.0)
    _lr622 = _lw622.groupby([df["date"], df["ind_code"]]).rank(pct=True)
    _mkw = (_lr622 > 0.8).astype(float)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
b = S561.replace(OLD, LEAD, 1).replace(OLD_DOC,
    "行业内龙头领先滞后（同伴 = 申万一级行业内前一日流通市值前 20% 的等权滞后收益，残差取负；暴露表错一天）", 1)
wr("622_单独行业内龙头领先滞后.py", b)

# ---- 623/624：按前一日风格暴露五分位分组的条件反转（换掉 c_condrev 的分组维度，并把它设为信号） ----
COND_OLD = '''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
assert COND_OLD in S561
for fname, col, desc in [
    ("623_单独按BETA五分位分组的条件反转.py", "s_BETA",
     "按前一日 BETA 五分位分组的条件反转（第 t 个截面上把多尺度反转排名在 BETA 五分位内去均值；暴露表错一天）"),
    ("624_单独按SIZE五分位分组的条件反转.py", "s_SIZE",
     "按前一日 SIZE 五分位分组的条件反转（第 t 个截面上把多尺度反转排名在 SIZE 五分位内去均值；暴露表错一天）"),
]:
    COND = f'''    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _qv = df["{col}"]
    _q66 = ((_qv.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))'''
    b = S561.replace(COND_OLD, COND, 1).replace(OLD_DOC, desc, 1)
    b = b.replace('    result["signal"] = result["c_indpeer"]', '    result["signal"] = result["c_condrev"]', 1)
    wr(fname, b)
