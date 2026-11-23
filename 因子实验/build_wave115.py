# -*- coding: utf-8 -*-
"""第一百一十五批：头部高权重组合（658 span5@0.45+条件0.15、659 span5@0.60、660 span5@0.40+条件0.15）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
# 658：基于 657（span5平滑同伴0.30+条件平滑0.15），把同伴权重提到 0.45
S657 = rd("657_头部加平滑同伴与条件反转平滑合成.py")
t = S657.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.15]", 1)
t = t.replace("span5 平滑同伴 0.30 + BETA 条件反转平滑版 0.15", "span5 平滑同伴 0.45 + BETA 条件反转平滑版 0.15", 1)
wr("658_头部加平滑同伴045与条件反转平滑015.py", t)
# 659：基于 589，span5 + 同伴权重 0.60
S589 = rd("589_头部加平滑同伴残差015.py")
t = S589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1).replace(
    "[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.60]", 1).replace(
    "10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。", "5 窗口平滑版 0.60。", 1)
wr("659_头部加平滑同伴残差span5060.py", t)
# 660：基于 637（头部+条件0.15），加 span5 平滑同伴 0.40
S637 = rd("637_头部加BETA条件反转015.py")
BLEND_OLD = 'rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_condrev"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15])'
BLEND_NEW = '''    _ord660 = result.sort_values(["instrument", "date"])
    result["c_ps660"] = _ord660.groupby("instrument", sort=False)["c_indpeer"].transform(
        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(result.index)
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ps660", "c_condrev"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.40, 0.15])'''
t = S637.replace("span=10, min_periods=5", "span=5, min_periods=3", 1).replace(BLEND_OLD, BLEND_NEW, 1)
lines = t.split("\n")
for i, ln in enumerate(lines):
    if ln.strip().startswith('"""') and ln.strip() != '"""':
        lines[i] = '    """合法头部 + span5 平滑同伴 0.40 + BETA 条件反转 0.15（暴露表错一天）。'
        break
wr("660_头部加平滑同伴040与条件反转015.py", "\n".join(lines))
