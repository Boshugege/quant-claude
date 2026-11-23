# -*- coding: utf-8 -*-
"""第九十五批：同伴残差家族的平滑化（低换手合格版）——把 575 的做法推广到小市值/全市场/换手加权三个同伴定义，
并做家族平滑合成与头部叠加。全部复用 497/561 的 SQL 文本。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
OLD = '''    result["signal"] = result["c_indpeer"]
    result["factor"] = result["signal"]'''
SMOOTH = '''    _ord95 = result.sort_values(["instrument", "date"])
    result["signal"] = _ord95.groupby("instrument", sort=False)["c_indpeer"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)
    result["factor"] = result["signal"]'''

for src, dst, desc in [
    ("585_单独行业内小市值加权同伴残差动量.py", "590_单独小市值同伴残差平滑版.py",
     "585（行业内小市值加权同伴残差动量）的 10 窗口 EWMA 平滑版：换手压到 0.2 档、与 104 高度分散"),
    ("586_单独全市场市值加权同伴残差动量.py", "591_单独全市场同伴残差平滑版.py",
     "586（全市场市值加权同伴残差动量）的 10 窗口 EWMA 平滑版"),
    ("587_单独行业内高换手加权同伴残差动量.py", "592_单独换手加权同伴残差平滑版.py",
     "587（行业内换手加权同伴残差动量）的 10 窗口 EWMA 平滑版"),
]:
    s = (CODE / src).read_text(encoding="utf-8")
    assert OLD in s, src
    b = s.replace(OLD, SMOOTH, 1)
    (CODE / dst).write_text(b, encoding="utf-8"); print("写出", dst)

# 593：家族平滑合成（四个同伴定义 + 平滑）——直接改 582（已含四定义、方向已校正）
s582 = (CODE / "582_单独多同伴定义合成反转.py").read_text(encoding="utf-8")
assert OLD in s582
b = s582.replace(OLD, SMOOTH, 1)
b = b.replace("四种同伴定义（行业内市值加权/小市值加权/前一日换手加权/行业×BETA三分位等权）的残差反转合成（方向已按 561/569 实测 IC 符号校正）；暴露表全部错一天",
              "四种同伴定义残差反转合成的 10 窗口 EWMA 平滑版（低换手、与 104 高度分散）")
(CODE / "593_单独多同伴定义合成平滑版.py").write_text(b, encoding="utf-8"); print("写出 593")

# 594：头部 + 593 的平滑合成信号 0.15
s573 = (CODE / "573_头部加行业内市值加权同伴残差015.py").read_text(encoding="utf-8")
old573 = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])'''
assert old573 in s573
new573 = '''    _ord94 = result.sort_values(["instrument", "date"])
    result["c_indpeer_s"] = _ord94.groupby("instrument", sort=False)["c_indpeer"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])'''
b = s573.replace(old573, new573, 1)
b = b.replace('    keep = ["r_30"', '    keep = ["c_indpeer_s", "r_30"', 1)
b = b.replace("合法头部 + 567 新源（行业内前一日市值加权同伴残差，方向已校正）0.15",
              "合法头部 + 567 新源平滑版 0.15（与 589 同，备用）")
(CODE / "594_头部加平滑同伴残差备.py").write_text(b, encoding="utf-8"); print("写出 594")
