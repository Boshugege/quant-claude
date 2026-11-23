# -*- coding: utf-8 -*-
"""第八十九批：567（合格新源）落地用法 —— 头部叠加两个剂量 + 该新源的低换手化版本。
全部复用 497/561 的 SQL 文本，评测机命中缓存。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S567 = (CODE / "567_单独行业内市值加权同伴残差动量.py").read_text(encoding="utf-8")
OLD = '''    result["signal"] = result["c_indpeer"]
    result["factor"] = result["signal"]'''
assert OLD in S567

def overlay(w):
    return '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, %s])
    result["factor"] = neutralize("signal", CHARS4, 1.5)''' % w

for fname, block, desc in [
    ("573_头部加行业内市值加权同伴残差015.py", overlay("0.15"),
     "合法头部 + 567 新源（行业内前一日市值加权同伴残差，方向已校正）0.15"),
    ("574_头部加行业内市值加权同伴残差030.py", overlay("0.30"),
     "合法头部 + 567 新源（行业内前一日市值加权同伴残差，方向已校正）0.30"),
]:
    body = S567.replace(OLD, block)
    body = body.replace("行业内前一日市值加权同伴残差动量（方向已按 561 实测 IC 符号校正：561 的 IC 为 −0.0765，本文件取正号）。",
                        desc + "。", 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)

# 575：把 567 新源的信号做 10 窗口 EWMA 平滑（只降换手，不改信息源）
SMOOTH = '''    _ord89 = result.sort_values(["instrument", "date"])
    _ew89 = _ord89.groupby("instrument", sort=False)["c_indpeer"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean())
    result["signal"] = _ew89.reindex(result.index)
    result["factor"] = result["signal"]'''
body = S567.replace(OLD, SMOOTH)
body = body.replace("行业内前一日市值加权同伴残差动量（方向已按 561 实测 IC 符号校正：561 的 IC 为 −0.0765，本文件取正号）。",
                    "567 新源的 10 窗口 EWMA 平滑版（只压换手，信息源不变）。", 1)
(CODE / "575_单独行业内市值加权同伴残差平滑版.py").write_text(body, encoding="utf-8")
print("写出 575_单独行业内市值加权同伴残差平滑版.py")
