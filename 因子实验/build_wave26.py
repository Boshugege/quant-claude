# -*- coding: utf-8 -*-
"""第三十批（261+）：在 250（新高）骨架上按任务说明第 5.4 节扩展中性化特征集，并试 EWMA 上影线。

261 = 250 + MAX 进中性化特征集（七特征）；
262 = 250 + 价格水平非线性项（c_lowpx²）进特征集；
263 = 250 的上影线改 EWMA 近端加权（把 246 的杠杆搬到新骨架）。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "250_长窗口六特征中性加上影线三成.py"

def write(num, name, text):
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(text, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)

def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:70], t.count(old))
    return t.replace(old, new, 1)

s0 = BASE.read_text(encoding="utf-8")
CH = '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"]\n'
KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh"]\n'

# 261: 加 MAX
s = repl(s0, CH, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2", "c_max"]\n', "261")
write("261", "长窗口七特征中性加上影线三成", s)

# 262: 加价格水平非线性项
s = repl(s0, '    df["c_lowpx"] = -np.log(px.where(px > 0))\n',
            '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'
            '    df["c_lowpx2"] = df["c_lowpx"] ** 2\n', "262a")
s = repl(s, KEEP, KEEP.replace('"c_upsh"]', '"c_upsh", "c_lowpx2"]'), "262b")
s = repl(s, CH, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2", "c_lowpx2"]\n', "262c")
write("262", "长窗口七特征含价格平方加上影线三成", s)

# 263: EWMA 上影线
s = repl(s0, '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n',
            '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n'
            '    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=20, min_periods=10).mean())\n', "263")
write("263", "长窗口六特征中性加上影线EWMA", s)
