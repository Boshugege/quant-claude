# -*- coding: utf-8 -*-
"""第一百三十批（把已取证的 731 新源并入头部）：
    754 = 合法头部（688 配方）但条件反转分组扩到 9 个风格（含 EARNYILD/GROWTH/LEVERAGE/BTOP）
    755 = 全 9 风格条件反转纯源（取证版，独立 IC_IR 与相关筛查用；无头部叠加、无中性化）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

HEAD = (CODE / "688_头部加span5同伴045与多风格条件反转020.py").read_text(encoding="utf-8")
OLD5 = '    for _col670 in ["s_BETA", "s_SIZE", "s_LIQUIDTY", "s_RESVOL", "s_MOMENTUM"]:'
NEW9 = '    for _col670 in ["s_BETA", "s_SIZE", "s_LIQUIDTY", "s_RESVOL", "s_MOMENTUM", "s_EARNYILD", "s_GROWTH", "s_LEVERAGE", "s_BTOP"]:'
assert OLD5 in HEAD, "688 分组循环未找到"
S754 = HEAD.replace(OLD5, NEW9, 1)
S754 = S754.replace("合法头部 + span5 平滑同伴 0.45 + 多风格条件反转平滑 0.20（暴露表错一天）。",
                    "合法头部（688 配方）但条件反转分组扩到 9 个风格（BETA/SIZE/LIQUIDTY/RESVOL/MOMENTUM/EARNYILD/GROWTH/LEVERAGE/BTOP）；暴露表错一天。", 1)
(CODE / "754_头部全九风格条件反转.py").write_text(S754, encoding="utf-8"); print("写出 754")

OLD_SIG = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert OLD_SIG in S754, "头部信号块未找到"
NEW_SIG = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
S755 = S754.replace(OLD_SIG, NEW_SIG, 1)
S755 = S755.replace("合法头部（688 配方）但条件反转分组扩到 9 个风格（BETA/SIZE/LIQUIDTY/RESVOL/MOMENTUM/EARNYILD/GROWTH/LEVERAGE/BTOP）；暴露表错一天。",
                    "全 9 风格（BETA/SIZE/LIQUIDTY/RESVOL/MOMENTUM/EARNYILD/GROWTH/LEVERAGE/BTOP）五分位内条件反转（纯源取证版，无头部叠加、无中性化）；暴露表错一天。", 1)
(CODE / "755_全九风格条件反转纯源.py").write_text(S755, encoding="utf-8"); print("写出 755")
