# -*- coding: utf-8 -*-
"""第十三批（195+）：用成交额平方加权（大单近似）重构 VWAP 成本锚阶梯。

与 195/196 之前的「成本锚」尝试不同：当时用**单笔成交规模**加权（需要逐笔数据，本地近似失真），
这里用**分钟成交额的平方**加权，在 1 分钟数据上完全可得，且天然放大成交量最大的那些分钟。
全部含 09:30 修复。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import keep, doc  # noqa: E402
from build_wave10 import base  # noqa: E402
from build_micro import add_minute_sql as micro_sql  # noqa: E402

MRET2_DN = """            CASE
                WHEN px_open > 0 AND px_close > 0 AND px_close < px_open
                THEN POWER(px_close / px_open - 1, 2) ELSE 0
            END AS mret2_dn,"""

MRET2_DN_ADD = MRET2_DN + """
            CASE WHEN px_close > 0 THEN amt * amt * px_close ELSE 0 END AS amt2_px,"""

WIN_OLD = """            SUM(mret2) AS sum_mret2,
            SUM(mret2_dn) AS sum_mret2_dn"""
WIN_NEW = WIN_OLD + """,
            SUM(amt2_px) AS sum_amt2_px"""

SEL_OLD = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close,\n"
           "        f.sum_amt2, f.max_min_amt, f.sum_amt_mn, f.sum_mret2, f.sum_mret2_dn")
SEL_NEW = SEL_OLD + ",\n        f.sum_amt2_px"

FLT_OLD = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
           '              "sum_amt2", "max_min_amt", "sum_amt_mn", "sum_mret2", "sum_mret2_dn"]:')
FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
           '              "sum_amt2", "max_min_amt", "sum_amt_mn", "sum_mret2", "sum_mret2_dn",\n'
           '              "sum_amt2_px"]:')


def add_bigvw_sql(src):
    src = micro_sql(src)
    assert src.count(MRET2_DN) == 1
    src = src.replace(MRET2_DN, MRET2_DN_ADD)
    assert src.count(WIN_OLD) == 1
    src = src.replace(WIN_OLD, WIN_NEW)
    assert src.count(SEL_OLD) == 1
    src = src.replace(SEL_OLD, SEL_NEW)
    assert src.count(FLT_OLD) == 1
    return src.replace(FLT_OLD, FLT_NEW)


SNIP_ADD_SCALE = '''
    # 大单成本锚：把 VWAP 的成交额权改成成交额平方权（分钟成交额越大权重越高），得到「大单主导的均价」，
    # 再按 4/8 个窗口累计，作为阶梯的两个新尺度（r_bv4 / r_bv8），与等权 VWAP 尺度并存。
    def rolling_bigvw_rev(k):
        _w = roll(df["sum_amt2"], k, "sum", 1)
        _wx = roll(df["sum_amt2_px"], k, "sum", 1)
        return safe_log_ratio(_wx / _w.where(_w > 0), px)

    df["r_bv4"] = rolling_bigvw_rev(4).fillna(r_dvwap)
    df["r_bv8"] = rolling_bigvw_rev(8).fillna(r_dvwap)
'''

SNIP_REPLACE_SCALE = '''
    # 大单成本锚（替换式）：把阶梯里的 4/8 窗口 VWAP 尺度整体换成成交额平方加权的「大单均价」，
    # 只保留同样数量的尺度，用来判断成本锚本身是否应该只看大单分钟。
    def rolling_bigvw_rev(k):
        _w = roll(df["sum_amt2"], k, "sum", 1)
        _wx = roll(df["sum_amt2_px"], k, "sum", 1)
        return safe_log_ratio(_wx / _w.where(_w > 0), px)

    df["r_rv4"] = rolling_bigvw_rev(4).fillna(r_dvwap)
    df["r_rv8"] = rolling_bigvw_rev(8).fillna(r_dvwap)
'''

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'
LADDER_OLD = '''        for c in ["r_30", "r_rv4", "r_rv8", "r_day"]:'''
LADDER_NEW = '''        for c in ["r_30", "r_rv4", "r_rv8", "r_bv4", "r_bv8", "r_day"]:'''

SPECS = [
    ("因子代码/195_大单成本锚阶梯.py", "add", """软同伴残差阶梯（长窗口 240 日）+ 大单成本锚两个新尺度 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    新信息：把 VWAP 的成交额权重换成**成交额平方**权重，得到「大单主导的均价」，按 4/8 个窗口累计后
      作为阶梯的新尺度（与等权 VWAP 尺度并存）。此前「机构成本锚」用单笔成交规模加权，结论是弱于普通 VWAP；
      本因子改用分钟成交额平方加权，在 1 分钟数据上不做任何分布假设。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/196_大单成本锚替换阶梯.py", "replace", """软同伴残差阶梯（长窗口 240 日）+ 大单成本锚**替换**等权 VWAP 尺度 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    与 195 号同源，但把阶梯里的 4/8 窗口 VWAP 尺度整体换成大单均价（尺度数量不变），
      用来直接比较「等权成本锚」与「大单成本锚」哪一个才是真正有信息的那一个。
    09:30 修复：同 104 号对照版本。"""),
]

if __name__ == "__main__":
    for path, mode, text in SPECS:
        s = base()
        s = add_bigvw_sql(s)
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + (SNIP_ADD_SCALE if mode == "add" else SNIP_REPLACE_SCALE))
        if mode == "add":
            assert s.count(LADDER_OLD) == 1
            s = s.replace(LADDER_OLD, LADDER_NEW)
            s = keep(s, ("r_bv4", "r_bv8"))
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
