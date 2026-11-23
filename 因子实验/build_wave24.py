# -*- coding: utf-8 -*-
"""第二十九批（253+）：上影线的"同伴残差化"（新方向）与窗口长度扫描。

253/254 = 上影线占比先做同伴残差（减去同月软同伴的上影线均值），再看单独使用/与原始上影线叠加是否更好。
255 = 上影线窗口从 40 延长到 60。全部基于 233 骨架（长窗口同伴阶梯 + 上影线，09:30 置零）。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "233_长上影线附加三成.py"

def write(num, name, text):
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(text, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)

def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:70], t.count(old))
    return t.replace(old, new, 1)

s0 = BASE.read_text(encoding="utf-8")
LADDER = '    result["ladder"] = peer_ladder()\n'
SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n'
RESID = ('    result["c_upsh_p"] = result["c_upsh"] - peer_mean("c_upsh").fillna(result.groupby("date")["c_upsh"].transform("mean"))\n')

# 253: 上影线换成同伴残差版
s = repl(s0, LADDER, LADDER + RESID, "253a")
s = repl(s, SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh_p"], [1.0, 0.30])\n', "253b")
write("253", "长上影线同伴残差三成", s)

# 254: 原始上影线 + 同伴残差上影线（小权重）
s = repl(s0, LADDER, LADDER + RESID, "254a")
s = repl(s, SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_upsh_p"], [1.0, 0.30, 0.15])\n', "254b")
write("254", "长上影线加同伴残差上影线", s)

# 255: 上影线窗口 60
s = repl(s0, '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n',
            '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 60, "mean", 10)\n', "255")
write("255", "长上影线窗口六十", s)
