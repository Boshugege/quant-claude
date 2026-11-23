# -*- coding: utf-8 -*-
"""第一百一十九批：span5/span7 平滑同伴 + 多风格条件反转的头部组合（673/675）与独立合成（674）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
S670 = rd("670_头部加平滑同伴与多风格条件反转合成.py")
# 673：把 c_indpeer_s 的 span10 改成 span5
t = S670.replace("span=10, min_periods=5", "span=5, min_periods=3", 1)
t = t.replace("span10 平滑同伴 0.30 + 多风格条件反转平滑 0.15", "span5 平滑同伴 0.30 + 多风格条件反转平滑 0.15", 1)
wr("673_头部加span5同伴与多风格条件反转.py", t)
# 675：span7 版
t = S670.replace("span=10, min_periods=5", "span=7, min_periods=4", 1)
t = t.replace("span10 平滑同伴 0.30 + 多风格条件反转平滑 0.15", "span7 平滑同伴 0.30 + 多风格条件反转平滑 0.15", 1)
wr("675_头部加span7同伴与多风格条件反转.py", t)
# 674：独立合成源 —— span5 平滑同伴 + 多风格条件反转 排名合成
OLD = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15])'''
NEW = '''    _rp674 = (result.groupby("date")["c_indpeer_s"].rank(pct=True) - 0.5)
    _rc674 = (result.groupby("date")["c_conds670"].rank(pct=True) - 0.5)
    result["signal"] = -_rp674 + _rc674'''
assert OLD in S670
t = S670.replace("span=10, min_periods=5", "span=5, min_periods=3", 1).replace(OLD, NEW, 1)
lines = t.split("\n")
for i, ln in enumerate(lines):
    if ln.strip().startswith('"""') and ln.strip() != '"""':
        lines[i] = '    """独立合成源：span5 平滑同伴 + 多风格条件反转 排名合成（暴露表错一天）。'
        break
wr("674_单独span5同伴与多风格条件反转合成.py", "\n".join(lines))
