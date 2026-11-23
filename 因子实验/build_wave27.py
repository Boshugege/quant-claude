# -*- coding: utf-8 -*-
"""第三十一批（264+）：把 258 验证有效的"平均单笔成交额"新数据轴搬到 250（新高）骨架上。

258（233 骨架 + 平均单笔成交额 0.15）拿到 +0.0693/z=3.25，与 233 等价；负号版 259 只有 +0.030/z=1.7，
说明"平均单笔成交额偏大"这一侧确有信息。264 = 250 + 同杠杆，检验它能否叠加到 250 上。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "250_长窗口六特征中性加上影线三成.py"

def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:70], t.count(old))
    return t.replace(old, new, 1)

s = BASE.read_text(encoding="utf-8")
s = repl(s, "            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt\n",
            "            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,\n"
            "            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals\n", "raw")
s = repl(s, "            px_low,\n",
            "            px_low,\n            deals,\n", "pass")
s = repl(s, "            SUM(upsh_c) AS sum_upsh\n",
            "            SUM(upsh_c) AS sum_upsh,\n"
            "            SUM(deals) AS sum_deals\n", "win")
s = repl(s, "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh\n",
            "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n", "sel")
s = repl(s, 'for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh"]:',
            'for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:', "cast")
s = repl(s, '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh"]\n',
            '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr"]\n', "keep")
s = repl(s, '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n',
            '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
            '    # 平均单笔成交额：Σ成交额/Σ委托笔数 的 40 窗口均值（大单/小单主导窗口的代理）。\n'
            '    _avgtr = df["sum_amt"] / df["sum_deals"].where(df["sum_deals"] > 0)\n'
            '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n', "comp")
s = repl(s, '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n',
            '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr"], [1.0, 0.30, 0.15])\n', "sig")
p = D / "264_长窗口六特征中性上影线加平均单笔成交额.py"
p.write_text(s, encoding="utf-8")
subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
print("写出", p)
