# -*- coding: utf-8 -*-
"""第一百零五批：头部 + span5 平滑同伴 0.45（634）+ 头部 + span7 平滑同伴 0.30（635），补 span/权重网格。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S589 = (CODE / "589_头部加平滑同伴残差015.py").read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
# 634：span5 + 0.45
t = S589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1)
t = t.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.45]", 1)
t = t.replace("10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。", "5 窗口平滑版 0.45。", 1)
wr("634_头部加平滑同伴残差span5045.py", t)
# 635：span7 + 0.30
t = S589.replace("span=10, min_periods=5", "span=7, min_periods=4", 1)
t = t.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.30]", 1)
t = t.replace("10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。", "7 窗口平滑版 0.30。", 1)
wr("635_头部加平滑同伴残差span7030.py", t)
