# -*- coding: utf-8 -*-
"""第二十八批（249+）：两条主线。

A. 把已验证的强杠杆组合到更强的骨架上：
   249 = 212（长窗口六尺度阶梯 + 上影线 0.15）把上影线提到 0.30；
   250 = 227（长窗口同伴阶梯 + 六特征中性化含换手/流通市值，无微结构杠杆）+ 上影线 0.30。
B. 新方向（与 104 截面相关更低的独立数据源）：盘口委托量不平衡。
   251/252 = 233 骨架 + 窗口内 Σ(买一量-卖一量)/Σ(买一量+卖一量)，正/负号各一个（符号未知，两个方向都试）。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"

def write(num, name, text):
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(text, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)

def repl(text, old, new, num):
    assert text.count(old) == 1, (num, old[:70], text.count(old))
    return text.replace(old, new, 1)

# ---- 249: 212 的上影线权重 0.15 -> 0.30
s = (D / "212_长窗口同伴上影线六尺度0930置零.py").read_text(encoding="utf-8")
s = repl(s, '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.15])\n',
            '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n', "249")
s = s.replace("长窗口同伴上影线六尺度（09:30 置零版）", "长窗口同伴上影线六尺度、上影线权重 0.30（09:30 置零版）", 1)
write("249", "长窗口六尺度上影线三成", s)

# ---- 250: 227（六特征中性化的纯阶梯）+ 上影线 0.30
s = (D / "227_长窗口同伴五特征中性0930置零.py").read_text(encoding="utf-8")
UPSH = '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
# 先把上影线需要的 SQL 列补进 227 的骨架
s = repl(s, "            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,\n",
            "            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,\n"
            "            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,\n", "250raw")
s = repl(s, "            pre_close,\n",
            "            pre_close,\n            px_high,\n            px_low,\n", "250pass")
s = repl(s, "            END AS px,\n",
            "            END AS px,\n"
            "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - px_low ELSE 0 END AS rng_c,\n"
            "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,\n", "250snap")
s = repl(s, "            MAX(pre_close) AS pre_close\n",
            "            MAX(pre_close) AS pre_close,\n"
            "            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,\n"
            "            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low,\n"
            "            SUM(rng_c) AS sum_rng,\n"
            "            SUM(upsh_c) AS sum_upsh\n", "250win")
s = repl(s, "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close\n",
            "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh\n", "250sel")
s = repl(s, 'for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:',
            'for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh"]:', "250cast")
s = repl(s, '    df["c_max"] = -roll(win_ret, 40, "max", 10)\n',
            '    df["c_max"] = -roll(win_ret, 40, "max", 10)\n'
            '    # 分钟上影线占比：窗口内 Σ(最高-实体上沿)/Σ(最高-最低)，40 窗口均值取负（冲高回落越多，反转越强）。\n' + UPSH, "250")
s = repl(s, '    result["factor"] = neutralize("ladder", CHARS4, 1.5)\n',
            '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n'
            '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n', "250")
s = repl(s, '"c_max", "c_turn", "c_mcap2"]', '"c_max", "c_turn", "c_mcap2", "c_upsh"]', "250")
write("250", "长窗口六特征中性加上影线三成", s)

# ---- 251/252: 233 + 盘口委托量不平衡（正/负号）
base = (D / "233_长上影线附加三成.py").read_text(encoding="utf-8")
s = repl(base, "            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,\n",
               "            CAST(COALESCE(ask_price1, 0) AS DOUBLE) AS ap1,\n"
               "            CAST(COALESCE(bid_volume1, 0) AS DOUBLE) AS bv1,\n"
               "            CAST(COALESCE(ask_volume1, 0) AS DOUBLE) AS av1,\n", "251sql")
s = repl(s, '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n',
            '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
            '            CASE WHEN bv1 > 0 OR av1 > 0 THEN bv1 ELSE 0 END AS ob_num,\n'
            '            CASE WHEN bv1 > 0 OR av1 > 0 THEN (bv1 + av1) ELSE 0 END AS ob_den,\n', "251snap")
s = repl(s, '            SUM(body_c) AS sum_body\n',
            '            SUM(body_c) AS sum_body,\n'
            '            SUM(ob_num) AS sum_obn,\n'
            '            SUM(ob_den) AS sum_obd\n', "251win")
s = repl(s, '        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body\n',
            '        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_obn, f.sum_obd\n', "251sel")
s = repl(s, '              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body"]:',
            '              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_obn", "sum_obd"]:', "251cast")
s = repl(s, '"c_max", "c_upsh"]', '"c_max", "c_upsh", "c_ob"]', "251keep")
for num, sign, nm in (("251", "+", "长上影线加盘口不平衡"), ("252", "-", "长上影线减盘口不平衡")):
    t = repl(s, '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n',
                '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
                '    # 盘口委托量不平衡：窗口内 Σ(买一量-卖一量)/Σ(买一量+卖一量) 的 40 窗口均值。\n'
                '    df["c_ob"] = %sroll(df["sum_obn"] / df["sum_obd"].where(df["sum_obd"] > 0), 40, "mean", 10)\n' % sign, num)
    t = repl(t, '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n',
                '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_ob"], [1.0, 0.30, 0.15])\n', num)
    write(num, nm, t)

if __name__ == "__main__":
    pass
