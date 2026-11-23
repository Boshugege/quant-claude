# -*- coding: utf-8 -*-
"""第六批（153+）：分钟 OHLC 形态（影线、实体、摆动比率、区间位置），全部含 09:30 修复。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, keep, tail, doc  # noqa: E402
from build_wave4 import soft  # noqa: E402

OK = "px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0"

SQL_SNAP_AFTER = "            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,"
SQL_SNAP_ADD = """
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,"""

SQL_SNAP_TAIL = """                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,"""
SQL_SNAP_TAIL_ADD = SQL_SNAP_TAIL + """
            CASE WHEN %(ok)s THEN px_high - px_low ELSE 0 END AS rng_c,
            CASE WHEN %(ok)s THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,
            CASE WHEN %(ok)s THEN LEAST(px_open, px_close) - px_low ELSE 0 END AS dnsh_c,
            CASE WHEN %(ok)s THEN ABS(px_close - px_open) ELSE 0 END AS body_c,""" % {"ok": OK}

SQL_WIN_OLD = "            MAX(pre_close) AS pre_close"
SQL_WIN_ADD = """
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low,
            SUM(rng_c) AS sum_rng,
            SUM(upsh_c) AS sum_upsh,
            SUM(dnsh_c) AS sum_dnsh,
            SUM(body_c) AS sum_body"""

SQL_SEL_OLD = "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close"
SQL_SEL_NEW = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n"
               "        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body")
SQL_FLT_OLD = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:'
SQL_FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n'
               '              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body"]:')

SNAP_SEL_OLD = """            trading_day,
            hms,
            pre_close,"""
SNAP_SEL_NEW = """            trading_day,
            hms,
            pre_close,
            px_high,
            px_low,"""

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

SNIPS = {
    "c_upsh": '''
    # 分钟上影线占比：窗口内 Σ(最高-实体上沿) / Σ(最高-最低)，最近 40 窗口均值取负（冲高回落越多，反转越强）。
    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)
''',
    "c_dnsh": '''
    # 分钟下影线占比：窗口内 Σ(实体下沿-最低) / Σ(最高-最低)，最近 40 窗口均值取正（砸盘被接回越多，后续越强）。
    df["c_dnsh"] = roll(df["sum_dnsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)
''',
    "c_shadow": '''
    # 影线不对称：(上影线 - 下影线) / 区间，最近 40 窗口均值取负。
    df["c_shadow"] = -roll((df["sum_upsh"] - df["sum_dnsh"]) / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)
''',
    "c_body": '''
    # 分钟实体占比：窗口内 Σ|收盘-开盘| / Σ(最高-最低)，最近 40 窗口均值取负（单边推动越强越易回吐）。
    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)
''',
    "c_path": '''
    # 摆动比率：Σ(分钟最高-最低) / (窗口最高-窗口最低)，度量窗口内价格来回震荡的程度，取负。
    _path = df["sum_rng"] / (df["win_high"] - df["win_low"]).where(df["win_high"] > df["win_low"])
    df["c_path"] = -roll(_path, 40, "mean", 10)
''',
    "c_pathpos": '''
    # 摆动比率的正号对照（其余完全一致），用于确认符号方向。
    _path = df["sum_rng"] / (df["win_high"] - df["win_low"]).where(df["win_high"] > df["win_low"])
    df["c_pathpos"] = roll(_path, 40, "mean", 10)
''',
    "c_pos": '''
    # 窗口末价在窗口区间中的位置 (末价-最低)/(最高-最低)，越靠上说明收在高位，取负。
    _pos = (df["last_px"] - df["win_low"]) / (df["win_high"] - df["win_low"]).where(df["win_high"] > df["win_low"])
    df["c_pos"] = -roll(_pos, 40, "mean", 10)
''',
}


def patch_sql(src):
    assert src.count(SQL_SNAP_AFTER) == 1
    src = src.replace(SQL_SNAP_AFTER, SQL_SNAP_AFTER + SQL_SNAP_ADD)
    assert src.count(SNAP_SEL_OLD) == 1
    src = src.replace(SNAP_SEL_OLD, SNAP_SEL_NEW)
    assert src.count(SQL_SNAP_TAIL) == 1
    src = src.replace(SQL_SNAP_TAIL, SQL_SNAP_TAIL_ADD)
    assert src.count(SQL_WIN_OLD) == 1
    src = src.replace(SQL_WIN_OLD, SQL_WIN_OLD + "," + SQL_WIN_ADD)
    assert src.count(SQL_SEL_OLD) == 1
    src = src.replace(SQL_SEL_OLD, SQL_SEL_NEW)
    assert src.count(SQL_FLT_OLD) == 1
    return src.replace(SQL_FLT_OLD, SQL_FLT_NEW)


SPECS = [
    ("因子代码/153_分钟上影线占比附加.py", "c_upsh", """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 分钟上影线占比附加（新方向，09:30 修复版）。

    第一次使用**分钟 bar 的 OHLC 形态**：窗口内「Σ(分钟最高价 - 分钟实体上沿) ÷ Σ(分钟最高-最低)」，
      即上影线占比，取负。依据：长上影线是无量突破/冲高回落的直接痕迹，代表上方抛压（海通金工
      「上下影线因子」一类的高频形态因子），与只看收盘价位置的 VWAP 类信号不是同一个量。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/154_分钟影线不对称附加.py", "c_shadow", """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 分钟影线不对称附加（新方向，09:30 修复版）。

    信号 = -((Σ上影线 - Σ下影线) ÷ Σ区间) 的 40 窗口均值：上影线相对下影线越多，说明推高被反复打回、
      买盘质量差，后续反转越强。与 153/下影线版本构成形态方向的正面检验。09:30 修复：同 104 号对照版本。"""),
    ("因子代码/155_分钟实体占比附加.py", "c_body", """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 分钟实体占比附加（新方向，09:30 修复版）。

    信号 = -(Σ|分钟收盘-分钟开盘| ÷ Σ(分钟最高-最低)) 的 40 窗口均值：实体占比高说明价格被单边推动、
      换手伴随方向性冲击，属于过度反应形态，后续越容易回吐。09:30 修复：同 104 号对照版本。"""),
    ("因子代码/156_摆动比率附加.py", "c_path", """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 摆动比率附加（新方向，09:30 修复版）。

    信号 = -(Σ(分钟最高-最低) ÷ (窗口最高-窗口最低)) 的 40 窗口均值：分子是窗口内价格走过的总路程，
      分母是净位移，比值大说明价格在同一区间来回震荡（流动性提供者活跃、方向性冲击小）。
    这是 Nagel (2012)「短期反转是提供流动性的报酬」的一个直接刻画，此前所有因子都没有用到价格路径长度。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/157_摆动比率正号对照.py", "c_pathpos", """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 摆动比率（**正号对照**，09:30 修复版）。

    与 156 号完全相同，只把符号取正，用于确认摆动比率到底是「震荡后反转更强」还是「震荡是流动性充足的
      表现、随后反转更弱」。同一时间提交两版，用真实评分直接定符号，比在本地猜更省时间。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/158_窗口区间位置附加.py", "c_pos", """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 窗口区间位置附加（新方向，09:30 修复版）。

    信号 = -(窗口末价在窗口区间内的位置) 的 40 窗口均值：00061 号做过日线级别的区间位置反转，
      本因子把它下推到 30 分钟窗口粒度，并与阶梯的「相对 VWAP / 相对昨收」不同——它只看窗口自身的形态。
    09:30 修复：同 104 号对照版本。"""),
]

if __name__ == "__main__":
    for path, col, text in SPECS:
        s = soft(BASE, k=200, power=3.0)
        s = patch_sql(s)
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + SNIPS[col])
        s = keep(s, (col,))
        s = tail(s, (col,), (0.15,))
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
