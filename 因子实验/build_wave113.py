# -*- coding: utf-8 -*-
"""第一百一十三批：头部权重补点——span5平滑同伴0.35（653）、SIZE条件反转0.25（654）、SIZE条件反转平滑0.30（655）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
S589 = rd("589_头部加平滑同伴残差015.py")
t = S589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1).replace(
    "[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.35]", 1).replace(
    "10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。", "5 窗口平滑版 0.35。", 1)
wr("653_头部加平滑同伴残差span5035.py", t)
S638 = rd("638_头部加SIZE条件反转015.py")
t = S638.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.25]", 1).replace(
    "按前一日 SIZE 五分位分组的条件反转 0.15", "按前一日 SIZE 五分位分组的条件反转 0.25", 1)
wr("654_头部加SIZE条件反转025.py", t)
S649 = rd("649_头部加SIZE条件反转平滑015.py")
t = S649.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.30]", 1).replace(
    "条件反转平滑版 0.15", "条件反转平滑版 0.30", 1)
wr("655_头部加SIZE条件反转平滑030.py", t)
