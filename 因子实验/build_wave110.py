# -*- coding: utf-8 -*-
"""第一百一十批：645 的两个邻点——span5 版（646）与 SIZE 条件反转版（647）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S645 = (CODE / "645_头部加平滑同伴与条件反转合成.py").read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
DOC = '    """合法头部 + span10 平滑同伴 0.30 + 按 BETA 五分位分组的条件反转 0.15（两个新源同时叠加；暴露表错一天）。'
t = S645.replace("span=10, min_periods=5", "span=5, min_periods=3", 1).replace(DOC, DOC.replace("span10", "span5"), 1)
wr("646_头部加平滑同伴span5与条件反转合成.py", t)
t = S645.replace('_qv645 = df["s_BETA"]', '_qv645 = df["s_SIZE"]', 1).replace("BETA 五分位分组", "SIZE 五分位分组", 1)
wr("647_头部加平滑同伴与SIZE条件反转合成.py", t)
