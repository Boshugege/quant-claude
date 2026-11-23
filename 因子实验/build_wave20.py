# -*- coding: utf-8 -*-
"""第二十五批（234+）：上影线权重与中性化组合的扫描（基于 233 = 176 骨架 + 上影线 0.30，实测 +0.0695/z=3.46）。

233 是目前最强的 7 截面候选（综合百分位 0.8895）。它在 IC（+0.129）、IC_IR（−0.010）、多空（+0.156）、
压力（−0.012）、换手（+0.085）上比 226（上影线 0.15）更均衡，说明上影线权重还有一档空间。
本批只扫一个参数（权重），并配两档中性化力度做交叉，避免多维过拟合。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "因子代码/233_长上影线附加三成.py"
SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n'
FRAC = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'

SPECS = [
    ("234", "长上影线附加四成半", (SIG, SIG.replace("0.30]", "0.45]")), None),
    ("235", "长上影线附加六成", (SIG, SIG.replace("0.30]", "0.60]")), None),
    ("236", "长上影线附加二成二", (SIG, SIG.replace("0.30]", "0.22]")), None),
    ("237", "长上影线三成中性二倍", None, 2.0),
    ("238", "长上影线三成中性一又二", None, 1.2),
]

def main():
    s0 = BASE.read_text(encoding="utf-8")
    for num, name, sig, frac in SPECS:
        s = s0
        if sig:
            old, new = sig
            assert s.count(old) == 1, (num, old)
            s = s.replace(old, new, 1)
        if frac:
            assert s.count(FRAC) == 1
            s = s.replace(FRAC, FRAC.replace("1.5)", "%s)" % frac), 1)
        p = ROOT / ("因子代码/%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)

if __name__ == "__main__":
    main()
