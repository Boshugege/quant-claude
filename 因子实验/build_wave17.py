# -*- coding: utf-8 -*-
"""第二十二批（218–225）：211（四杠杆+09:30置零，已验证 +0.064/z=3.23）的参数族。

目标：211 的短板是 IC_IR 与压力（pct_diff 为负），长处是 IC、多空夏普与换手。
本批沿三条已知杠杆扫描：中性化力度（103→104 说明 1.0→1.5 提升 IC_IR/压力）、
中性化特征集（加 MAX）、同伴窗口与 k（104 的短窗硬同伴给出最高 IC_IR）。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "因子代码/211_四杠杆叠加0930置零.py"

FRAC = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
CHARS = '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]\n'
WEIGHTS = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_m30"], [1.0, 0.15, 0.15])\n'
LOOKBACK = '    LOOKBACK_DAYS = 400\n'
PEER = '    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 240, 120, 3.0\n'

SPECS = [
    ("218", "四杠杆中性二倍", [(FRAC, '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n')],
     "中性化力度 1.5 → 2.0（目标抬 IC_IR/压力）。"),
    ("219", "四杠杆中性二五倍", [(FRAC, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n')],
     "中性化力度 1.5 → 2.5。"),
    ("220", "四杠杆中性一点二倍", [(FRAC, '    result["factor"] = neutralize("signal", CHARS4, 1.2)\n')],
     "中性化力度 1.5 → 1.2（下界对照）。"),
    ("221", "四杠杆五特征中性", [(CHARS, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max"]\n')],
     "中性化特征加 MAX（c_max 已在代码里构造）。"),
    ("222", "四杠杆附加权重一成", [(WEIGHTS, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_m30"], [1.0, 0.10, 0.10])\n')],
     "两个附加项权重 0.15 → 0.10（降噪，目标抬 IC_IR）。"),
    ("223", "四杠杆附加权重两成", [(WEIGHTS, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_m30"], [1.0, 0.20, 0.20])\n')],
     "两个附加项权重 0.15 → 0.20。"),
    ("224", "四杠杆同伴k400窗400", [(PEER, '    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 400, 400, 200, 3.0\n'),
                                   (LOOKBACK, '    LOOKBACK_DAYS = 640\n')],
     "同伴 k=400、窗口 400 日（更长的相似性历史）。"),
    ("225", "四杠杆同伴k200窗400", [(PEER, '    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 400, 200, 3.0\n'),
                                   (LOOKBACK, '    LOOKBACK_DAYS = 640\n')],
     "同伴 k=200、窗口 400 日（只加长历史，不改广度）。"),
]

def main():
    s0 = BASE.read_text(encoding="utf-8")
    for num, name, subs, note in SPECS:
        s = s0
        for old, new in subs:
            assert s.count(old) == 1, (num, old)
            s = s.replace(old, new, 1)
        s = s.replace('    """四杠杆叠加：长窗口同伴 + 六尺度 + 分钟上影线 + 14:30 首半小时动量（09:30 置零版）。',
                      '    """四杠杆叠加（09:30 置零版）的%s。' % name, 1)
        s = s.replace('    """\n    import numpy as np', '\n    %s"""\n    import numpy as np' % note, 1)
        p = ROOT / ("因子代码/%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        print("写出", p)
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)

if __name__ == "__main__":
    main()
