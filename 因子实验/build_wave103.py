# -*- coding: utf-8 -*-
"""第一百零三批：分钟收益自相关（617，新机制）+ 30 分钟窗口自相关（618）+ 头部平滑同伴权重梯度（620/621）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)

S565 = rd("565_单独分钟价格路径持续性.py")

# 插入 1 分钟收益自协方差项（滞后 1）：Σ(Δpx_t · Δpx_{t-1}) 与其有效样本数
MARK_PAIR = '            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 AND (px - px_l1) <> 0 AND (px_l1 - px_l2) <> 0 THEN 1 ELSE 0 END) AS n_pppair,'
AC_ADD = (MARK_PAIR + '\n'
          '            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 THEN (px - px_l1) * (px_l1 - px_l2) ELSE 0 END) AS sum_ac,\n'
          '            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 THEN 1 ELSE 0 END) AS n_ac,')
SEL_OLD = '        f.sum_amt, f.sum_amt2, f.n_vpsync, f.n_vppair, f.n_ppsame, f.n_pppair, f.n_flat, f.n_trad, f.n_min,'
SEL_NEW = '        f.sum_amt, f.sum_amt2, f.n_vpsync, f.n_vppair, f.n_ppsame, f.n_pppair, f.sum_ac, f.n_ac, f.n_flat, f.n_trad, f.n_min,'
DT_OLD = 'for c in ["sum_amt", "sum_amt2", "n_vpsync", "n_vppair", "n_ppsame", "n_pppair", "n_flat", "n_trad", "n_min",'
DT_NEW = 'for c in ["sum_amt", "sum_amt2", "n_vpsync", "n_vppair", "n_ppsame", "n_pppair", "sum_ac", "n_ac", "n_flat", "n_trad", "n_min",'
assert MARK_PAIR in S565 and SEL_OLD in S565 and DT_OLD in S565
base = S565.replace(MARK_PAIR, AC_ADD, 1).replace(SEL_OLD, SEL_NEW, 1).replace(DT_OLD, DT_NEW, 1)

SIG_OLD = '''    df["c_ppsame"] = _rk(_smooth(_pp), -1.0)
    keep = ["c_ppsame"]'''
SIG_617 = '''    _ac = df["sum_ac"] / df["n_ac"].where(df["n_ac"] > 0)
    df["c_mac"] = _rk(_smooth(_ac), -1.0)   # 1 分钟收益的滞后 1 自相关（40 窗口均值，取负）
    keep = ["c_mac"]'''
assert SIG_OLD in base
b617 = base.replace(SIG_OLD, SIG_617, 1)
b617 = b617.replace('    """窗口内相邻分钟收益同号占比，40 窗口均值取负：路径越持续越偏趋势（09:30 置零版）。',
                    '    """分钟内相邻收益的滞后 1 自相关（ΣΔpx_t·Δpx_{t-1} / 有效样本数，40 窗口均值取负；09:30 置零版）。', 1)
b617 = b617.replace('    result["signal"] = result["c_ppsame"]', '    result["signal"] = result["c_mac"]', 1)
wr("617_单独分钟收益自相关.py", b617)

SIG_618 = '''    _ac = df["sum_ac"] / df["n_ac"].where(df["n_ac"] > 0)
    _absac = _ac.abs()
    df["c_macabs"] = _rk(_smooth(_absac), 1.0)   # |自相关|：趋势/震荡强度（40 窗口均值）
    keep = ["c_macabs"]'''
b618 = base.replace(SIG_OLD, SIG_618, 1)
b618 = b618.replace('    """窗口内相邻分钟收益同号占比，40 窗口均值取负：路径越持续越偏趋势（09:30 置零版）。',
                    '    """|分钟内相邻收益滞后 1 自相关|：趋势/震荡强度（40 窗口均值；09:30 置零版）。', 1)
b618 = b618.replace('    result["signal"] = result["c_ppsame"]', '    result["signal"] = result["c_macabs"]', 1)
wr("618_单独分钟收益自相关绝对值.py", b618)

# 头部 + span10 平滑同伴权重梯度（620=0.30、621=0.60）
S589 = rd("589_头部加平滑同伴残差015.py")
b620 = S589.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.30]", 1)
b620 = b620.replace("10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。",
                    "10 窗口平滑版 0.30（低换手、与 104 相关约 0.50 的新源）。", 1)
wr("620_头部加平滑同伴残差span1030.py", b620)

b621 = S589.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.60]", 1)
b621 = b621.replace("10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。",
                    "10 窗口平滑版 0.60（低换手、与 104 相关约 0.50 的新源）。", 1)
wr("621_头部加平滑同伴残差span1060.py", b621)
