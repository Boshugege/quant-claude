# -*- coding: utf-8 -*-
"""第一百零八批：头部权重网格细化（641 span5@0.40）+ 头部条件反转权重上探（642/643 @0.30）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
S589 = rd("589_头部加平滑同伴残差015.py")
t = S589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1)
t = t.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.40]", 1)
t = t.replace("10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。", "5 窗口平滑版 0.40。", 1)
wr("641_头部加平滑同伴残差span5040.py", t)
for src, dst, desc in [("637_头部加BETA条件反转015.py","642_头部加BETA条件反转030.py","BETA"),
                       ("638_头部加SIZE条件反转015.py","643_头部加SIZE条件反转030.py","SIZE")]:
    t = rd(src).replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.30]", 1)
    t = t.replace(f"按前一日 {desc} 五分位分组的条件反转 0.15", f"按前一日 {desc} 五分位分组的条件反转 0.30", 1)
    wr(dst, t)
