# -*- coding: utf-8 -*-
"""第一百一十二批：头部两源权重配比（650/652）+ 平滑同伴与条件反转的独立合成源（651）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
# 650：646 的 cond 权重 0.15→0.30
S646 = rd("646_头部加平滑同伴span5与条件反转合成.py")
t = S646.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.30]", 1)
t = t.replace("条件反转 0.15", "条件反转 0.30", 1)
wr("650_头部加平滑同伴span5与条件反转030合成.py", t)
# 652：头部 + span5 平滑同伴 0.20（下限邻点）
S589 = rd("589_头部加平滑同伴残差015.py")
t = S589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1).replace(
    "[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.20]", 1).replace(
    "10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。", "5 窗口平滑版 0.20。", 1)
wr("652_头部加平滑同伴残差span5020.py", t)
# 651：独立源 = 平滑同伴（span10）校正 与 风格条件反转 的排名合成
S623 = rd("623_单独按BETA五分位分组的条件反转.py")
OLD = '    result["signal"] = result["c_condrev"]'
NEW = ('    _ord651 = result.sort_values(["instrument", "date"])\n'
       '    result["_peers651"] = _ord651.groupby("instrument", sort=False)["c_indpeer"].transform(\n'
       '        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)\n'
       '    _rp651 = (result.groupby("date")["_peers651"].rank(pct=True) - 0.5)\n'
       '    _rc651 = (result.groupby("date")["c_condrev"].rank(pct=True) - 0.5)\n'
       '    result["signal"] = -_rp651 + _rc651')
t = S623.replace(OLD, NEW, 1).replace(
    "（第 t 个截面上把多尺度反转排名在 BETA 五分位内去均值；暴露表错一天）",
    " 与平滑后的同伴残差按排名合成（暴露表错一天）", 1)
wr("651_单独平滑同伴与条件反转合成.py", t)
