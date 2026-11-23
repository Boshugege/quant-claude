# -*- coding: utf-8 -*-
"""第一百三十九批（围绕本地最高估计点 766 做邻点，映射局部最优）：
    785 = 766 但条件反转分量权重 0.15
    786 = 766 但同伴残差分量权重 0.35
    787 = 766 但同伴残差分量用 span7 平滑
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S = (CODE / "766_头部换手角二点零中性化.py").read_text(encoding="utf-8")

B = '                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20])'
assert B in S
(CODE / "785_换手角源015二点零中性化.py").write_text(
    S.replace(B, '                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.40, 0.15])', 1), encoding="utf-8"); print("写出 785")
(CODE / "786_换手角同伴035二点零中性化.py").write_text(
    S.replace(B, '                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.35, 0.20])', 1), encoding="utf-8"); print("写出 786")

OLD7 = '''    result["c_indpeer_s"] = _ord89.groupby("instrument", sort=False)["c_indpeer"].transform(
        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(result.index)'''
assert OLD7 in S
(CODE / "787_换手角同伴span7二点零中性化.py").write_text(
    S.replace(OLD7, '''    result["c_indpeer_s"] = _ord89.groupby("instrument", sort=False)["c_indpeer"].transform(
        lambda x: x.ewm(span=7, min_periods=4).mean()).reindex(result.index)''', 1), encoding="utf-8"); print("写出 787")
