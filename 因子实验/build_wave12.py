# -*- coding: utf-8 -*-
"""第十二批（190+）：成交额结构（大单加权方向）与同伴定义的新维度（极端共跌依赖、相关性时间加权）。

前两个沿用「底座 + 单个新附加项（0.15）」的评测口径；后三个改动同伴的挑选/加权方式本身。
全部含 09:30 修复。
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import keep, tail, doc  # noqa: E402
from build_wave10 import base  # noqa: E402
from build_micro import add_minute_sql as micro_sql  # noqa: E402

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

MRET2_DN = """            CASE
                WHEN px_open > 0 AND px_close > 0 AND px_close < px_open
                THEN POWER(px_close / px_open - 1, 2) ELSE 0
            END AS mret2_dn,"""

MRET2_DN_ADD = MRET2_DN + """
            CASE WHEN px_open > 0 AND px_close > 0
                 THEN amt * (px_close / px_open - 1) ELSE 0 END AS amt_mret,
            CASE WHEN px_open > 0 AND px_close > 0
                 THEN amt * amt * (px_close / px_open - 1) ELSE 0 END AS amt2_mret,"""

WIN_OLD = """            SUM(mret2) AS sum_mret2,
            SUM(mret2_dn) AS sum_mret2_dn"""
WIN_NEW = WIN_OLD + """,
            SUM(amt_mret) AS sum_amt_mret,
            SUM(amt2_mret) AS sum_amt2_mret"""

SEL_OLD = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close,\n"
           "        f.sum_amt2, f.max_min_amt, f.sum_amt_mn, f.sum_mret2, f.sum_mret2_dn")
SEL_NEW = SEL_OLD + ",\n        f.sum_amt_mret, f.sum_amt2_mret"

FLT_OLD = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
           '              "sum_amt2", "max_min_amt", "sum_amt_mn", "sum_mret2", "sum_mret2_dn"]:')
FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
           '              "sum_amt2", "max_min_amt", "sum_amt_mn", "sum_mret2", "sum_mret2_dn",\n'
           '              "sum_amt_mret", "sum_amt2_mret"]:')

NEW_COLS = ["sum_amt_mret", "sum_amt2_mret"]


def add_bigw_sql(src):
    src = micro_sql(src)
    assert src.count(MRET2_DN) == 1
    src = src.replace(MRET2_DN, MRET2_DN_ADD)
    assert src.count(WIN_OLD) == 1
    src = src.replace(WIN_OLD, WIN_NEW)
    assert src.count(SEL_OLD) == 1
    src = src.replace(SEL_OLD, SEL_NEW)
    assert src.count(FLT_OLD) == 1
    return src.replace(FLT_OLD, FLT_NEW)


CONST_RE = re.compile(r"^(    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = .*\n)", re.M)
HIST_OLD = "        hist = daily_ret.loc[daily_ret.index < in_month[0]].tail(PEER_LOOKBACK)\n"


def add_halflife(src, hl):
    """同伴相关性改用「越近的交易日权重越高」的加权相关（指数半衰期）。"""
    assert src.count(HIST_OLD) == 1
    new = HIST_OLD + (
        "        _w = np.power(0.5, np.arange(len(hist))[::-1] / float(HIST_HALFLIFE))"
        "   # 近期样本权重更高\n"
        "        hist = hist.mul(_w, axis=0)\n")
    src = src.replace(HIST_OLD, new)
    assert CONST_RE.search(src)
    return CONST_RE.sub(lambda m: m.group(1) + "    HIST_HALFLIFE = %.1f\n" % hl, src, count=1)


CWEIGHT_OLD = """        cvals = corr[np.arange(len(corr))[:, None], top]
        wts = np.clip(cvals, 0.0, None) ** PEER_POWER   # 软同伴：正相关度加权"""
CWEIGHT_TAIL = """        cvals = corr[np.arange(len(corr))[:, None], top]
        # 极端共跌依赖：窗口内两只股票同时落在各自 10% 分位以下的频率协方差（经验尾部依赖），
        # 用它替代「正相关度」做加权，检验同伴的定义是否应该只看尾部共同风险。
        _h = hist.loc[:, ok]
        _q = _h.quantile(0.10).values
        _L = (_h.values < _q[None, :]).astype(float)
        _p = _L.mean(axis=0)
        _tailm = (_L.T @ _L) / _L.shape[0] - np.outer(_p, _p)
        _tw = _tailm[np.arange(len(corr))[:, None], top]
        wts = np.clip(_tw, 0.0, None)                    # 尾部依赖加权（弱正相关但强共跌的同伴权重更高）
        _fall = np.clip(cvals, 0.0, None) ** PEER_POWER   # 没有任何正尾部依赖时退回正相关度加权
        wts = np.where(wts.sum(axis=1, keepdims=True) > 0, wts, _fall)"""


def tail_weight(src):
    assert src.count(CWEIGHT_OLD) == 1
    return src.replace(CWEIGHT_OLD, CWEIGHT_TAIL)


SNIP_BIGW = '''
    # 大单加权方向偏离：窗口内「成交额平方加权的分钟收益」减「成交额加权的分钟收益」，最近 40 窗口均值取负。
    # 前者被大额分钟主导、后者被普通分钟主导，两者之差即「大单在往哪个方向推」。
    _bw = (df["sum_amt2_mret"] / df["sum_amt2"].where(df["sum_amt2"] > 0)
           - df["sum_amt_mret"] / df["sum_amt"].where(df["sum_amt"] > 0))
    df["c_bigw"] = -roll(_bw, 40, "mean", 10)
'''

SNIP_BIGWPOS = '''
    # 大单加权方向偏离的**正号对照**（其余完全一致），用来直接定方向。
    _bw = (df["sum_amt2_mret"] / df["sum_amt2"].where(df["sum_amt2"] > 0)
           - df["sum_amt_mret"] / df["sum_amt"].where(df["sum_amt"] > 0))
    df["c_bigwpos"] = roll(_bw, 40, "mean", 10)
'''

SPECS = [
    ("因子代码/190_大单加权方向偏离附加.py", "c_bigw", {}, """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 大单加权方向偏离附加（新方向，09:30 修复版）。

    新信息：窗口内「成交额² 加权的分钟收益」与「成交额加权的分钟收益」之差，40 窗口均值取负。
      不用窗口函数、不用逐笔方向，只用「大单分钟主导的方向」相对「普通分钟主导的方向」的偏离。
      这是开源证券「理想反转」类思路的可落地版本：大单时段的方向更可能被过度定价，之后的回吐更明显。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/191_大单加权方向偏离正号对照.py", "c_bigwpos", {}, """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 大单加权方向偏离（**正号对照**，09:30 修复版）。

    与 190 号只差符号。09:30 修复：同 104 号对照版本。"""),
    ("因子代码/192_极端共跌加权同伴.py", None, {"tail": True}, """极端共跌依赖加权的软同伴残差阶梯 + 四特征 1.5 倍中性化（同伴定义新维度，09:30 修复版）。

    改动同伴的**加权方式**：仍按相关性选出候选，但权重不再用正相关度，而是用「两只股票同时落在各自 10% 分位
      以下的频率协方差」（经验尾部依赖），即只奖励真正一起崩过的同伴（Ang & Chen 2002 的极端下行依赖）。
      这与「相关性高」不是一回事：慢慢同涨的股票相关但不共崩，而共同风险暴露体现在尾部。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/193_同伴相关半衰期十日.py", None, {"hl": 10.0}, """时间加权（半衰期 10 个交易日）的软同伴残差阶梯 + 四特征 1.5 倍中性化（同伴定义新维度，09:30 修复版）。

    改动同伴挑选的**时间权重**：把过去 240 个交易日的收益按指数半衰期 10 日加权后再算相关，
      即「共同成分的近端结构」权重更高。与 170（240 日等权）对照，检验同伴关系是稳定还是需要快速更新。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/194_同伴相关半衰期三十日.py", None, {"hl": 30.0}, """时间加权（半衰期 30 个交易日）的软同伴残差阶梯 + 四特征 1.5 倍中性化（同伴定义新维度，09:30 修复版）。

    与 193 号同源，只把半衰期从 10 日改成 30 日，两端夹逼出「时间权重」这个杠杆的最优区间。
    09:30 修复：同 104 号对照版本。"""),
]

if __name__ == "__main__":
    for path, col, opts, text in SPECS:
        s = base()
        if col is not None:
            s = add_bigw_sql(s)
            assert s.count(ANCHOR) == 1
            s = s.replace(ANCHOR, ANCHOR + (SNIP_BIGW if col == "c_bigw" else SNIP_BIGWPOS))
            s = keep(s, (col,))
            s = tail(s, (col,), (0.15,))
        if opts.get("tail"):
            s = tail_weight(s)
        if opts.get("hl"):
            s = add_halflife(s, opts["hl"])
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
