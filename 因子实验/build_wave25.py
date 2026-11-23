# -*- coding: utf-8 -*-
"""第二十九批（258+）：委托笔数 / 平均单笔成交额——任务说明第 5.6、5.7 节直接点名、但尚未在正确标签下重测的数据轴。

258/259 = 233 骨架 + 平均单笔成交额（Σ成交额/Σ委托笔数）的 40 窗口均值，正负号各一个；
260 = 233 骨架 + 分钟委托笔数（log）的 40 窗口均值。
数据轴与 104 的价格类信号正交，是"与 104 相关明显更低"的候选。
"""
import pathlib, subprocess, sys, numpy as np
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

# 给 233 骨架补上委托笔数列
s = BASE.read_text(encoding="utf-8")
s = repl(s, "            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt\n",
            "            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,\n"
            "            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals\n", "raw")
s = repl(s, "            px_low,\n",
            "            px_low,\n            deals,\n", "pass")
s = repl(s, "            SUM(body_c) AS sum_body\n",
            "            SUM(body_c) AS sum_body,\n"
            "            SUM(deals) AS sum_deals\n", "win")
s = repl(s, "        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body\n",
            "        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n", "sel")
s = repl(s, '              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body"]:',
            '              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:', "cast")
s = repl(s, '"c_max", "c_upsh"]', '"c_max", "c_upsh", "c_avgtr", "c_deals"]', "keep")
ANCHOR = '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
s = repl(s, ANCHOR, ANCHOR +
         '    # 平均单笔成交额：Σ成交额/Σ委托笔数，40 窗口均值（大小单主导的代理）。\n'
         '    _avgtr = df["sum_amt"] / df["sum_deals"].where(df["sum_deals"] > 0)\n'
         '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
         '    # 委托笔数（log）：窗口内成交笔数的 40 窗口均值，活跃度/散户参与度代理。\n'
         '    df["c_deals"] = roll(np.log1p(df["sum_deals"]), 40, "mean", 10)\n', "comp")
SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n'
assert s.count(SIG) == 1
for num, name, newsig in (
    ("258", "长上影线加平均单笔成交额", '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr"], [1.0, 0.30, 0.15])\n'),
    ("259", "长上影线减平均单笔成交额", '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr"], [1.0, 0.30, -0.15])\n'),
    ("260", "长上影线加委托笔数", '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_deals"], [1.0, 0.30, 0.15])\n'),
):
    write(num, name, s.replace(SIG, newsig, 1))
