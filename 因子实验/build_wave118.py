# -*- coding: utf-8 -*-
"""第一百一十八批：把 651 的合成源作为单一分量叠加头部（671）+ 多风格条件反转与平滑同伴的独立合成（672）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
# 671：头部 + （平滑同伴 - 多风格条件反转）合成分量 0.30
S670 = rd("670_头部加平滑同伴与多风格条件反转合成.py")
OLD = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15])'''
NEW = '''    _rp671 = (result.groupby("date")["c_indpeer_s"].rank(pct=True) - 0.5)
    _rc671 = (result.groupby("date")["c_conds670"].rank(pct=True) - 0.5)
    result["c_comp671"] = -_rp671 + 0.5 * _rc671
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_comp671"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.30])'''
assert OLD in S670
wr("671_头部加合成新源030.py", S670.replace(OLD, NEW, 1))
# 672：独立合成源 = 平滑同伴 + 多风格条件反转（均取平滑形态）
S670b = S670
OLD2 = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15])'''
NEW2 = '''    _rp672 = (result.groupby("date")["c_indpeer_s"].rank(pct=True) - 0.5)
    _rc672 = (result.groupby("date")["c_conds670"].rank(pct=True) - 0.5)
    result["signal"] = -_rp672 + _rc672'''
assert OLD2 in S670b
wr("672_单独平滑同伴与多风格条件反转合成.py", S670b.replace(OLD2, NEW2, 1))
