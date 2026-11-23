# -*- coding: utf-8 -*-
"""生成分钟级微结构方向的平台因子代码（123+，全部含 09:30 修复）。

所有候选都以 114 号（软同伴残差阶梯 + 四特征 1.5 倍中性化）为底座，只加一个新方向的附加项，
这样评分机结果可以直接读出「该新方向本身的增量」。附加项权重统一 0.15（与 121/122 同口径）。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, soft_peer, keep, tail, doc  # noqa: E402

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

# ---- SQL：把分钟级聚合列加进快照与窗口聚合 ----
SQL_SNAP_AFTER = "            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,"
SQL_SNAP_ADD = """
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,
            CAST(SUBSTR(hms, 1, 2) AS INTEGER) * 60 + CAST(SUBSTR(hms, 3, 2) AS INTEGER) AS mn,"""

SQL_PX_TAIL = """                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,"""
SQL_PX_TAIL_ADD = """                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,
            CASE
                WHEN vol > 0 AND amt > 0 AND px_low > 0
                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02
                THEN amt * amt ELSE 0
            END AS amt2_c,
            CASE WHEN px_open > 0 AND px_close > 0 THEN px_close / px_open - 1 ELSE 0 END AS mret,
            CASE WHEN px_open > 0 AND px_close > 0 THEN POWER(px_close / px_open - 1, 2) ELSE 0 END AS mret2,
            CASE
                WHEN px_open > 0 AND px_close > 0 AND px_close < px_open
                THEN POWER(px_close / px_open - 1, 2) ELSE 0
            END AS mret2_dn,"""

SQL_WIN_OLD = "            MAX(pre_close) AS pre_close"
SQL_WIN_ADD = """
            SUM(amt2_c) AS sum_amt2,
            MAX(amt_c) AS max_min_amt,
            SUM(amt_c * mn) AS sum_amt_mn,
            SUM(mret2) AS sum_mret2,
            SUM(mret2_dn) AS sum_mret2_dn"""

SQL_SEL_OLD = "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close"
SQL_SEL_NEW = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close,\n"
               "        f.sum_amt2, f.max_min_amt, f.sum_amt_mn, f.sum_mret2, f.sum_mret2_dn")

SQL_FLT_OLD = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:'
SQL_FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
               '              "sum_amt2", "max_min_amt", "sum_amt_mn", "sum_mret2", "sum_mret2_dn"]:')


SNAP_SEL_OLD = """            trading_day,
            hms,
            pre_close,"""
SNAP_SEL_NEW = """            trading_day,
            hms,
            pre_close,
            mn,"""


def add_minute_sql(src):
    assert src.count(SQL_SNAP_AFTER) == 1
    src = src.replace(SQL_SNAP_AFTER, SQL_SNAP_AFTER + SQL_SNAP_ADD)
    assert src.count(SNAP_SEL_OLD) == 1
    src = src.replace(SNAP_SEL_OLD, SNAP_SEL_NEW)
    assert src.count(SQL_PX_TAIL) == 1
    src = src.replace(SQL_PX_TAIL, SQL_PX_TAIL_ADD)
    assert src.count(SQL_WIN_OLD) == 1
    src = src.replace(SQL_WIN_OLD, SQL_WIN_OLD + "," + SQL_WIN_ADD)
    assert src.count(SQL_SEL_OLD) == 1
    src = src.replace(SQL_SEL_OLD, SQL_SEL_NEW)
    assert src.count(SQL_FLT_OLD) == 1
    return src.replace(SQL_FLT_OLD, SQL_FLT_NEW)


# ---- pandas：各新方向的分量 ----
SNIP_CONC = '''
    # 分钟成交额集中度（HHI）：窗口内 Σamt² / (Σamt)²，越大说明成交集中在少数分钟（大单冲击、过度反应）。
    _hhi = df["sum_amt2"] / df["sum_amt"].pow(2).where(df["sum_amt"] > 0)
    df["c_conc"] = -roll(_hhi, 40, "mean", 10)
'''

SNIP_MAXMIN = '''
    # 单分钟最大成交额占窗口成交额的比重（大单冲击的极值形态），取负：冲击越大，后续反转越强。
    _mx = df["max_min_amt"] / df["sum_amt"].where(df["sum_amt"] > 0)
    df["c_maxmin"] = -roll(_mx, 40, "mean", 10)
'''

SNIP_TILT = '''
    # 窗口内成交量的时点重心相对窗口中点（分钟）：>0 说明成交集中在窗口后段（买盘后段发力），取负。
    _mid = df["date"].dt.hour * 60 + df["date"].dt.minute
    _tilt = (df["sum_amt_mn"] / df["sum_amt"].where(df["sum_amt"] > 0) - _mid) / 15.0
    df["c_tilt"] = -roll(_tilt, 40, "mean", 10)
'''

SNIP_MDSD = '''
    # 1 分钟粒度下行波动占比：窗口内负分钟收益平方和 / 全部分钟收益平方和，取负。
    _mdsd = df["sum_mret2_dn"] / df["sum_mret2"].replace(0.0, np.nan)
    df["c_mdsd"] = -roll(_mdsd, 40, "mean", 10)
'''

SNIP_VSHOCK = '''
    # 相对放量：当日累计成交额 / 过去 5 个交易日全天成交额均值（只用过去数据），取负。
    _day_amt = df.groupby(["instrument", "day"], sort=False)["sum_amt"].sum().rename("d_amt").reset_index()
    _day_amt = _day_amt.sort_values(["instrument", "day"])
    _day_amt["prev5"] = _day_amt.groupby("instrument", sort=False)["d_amt"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).mean()
    )
    _pmap = _day_amt.set_index(["instrument", "day"])["prev5"]
    _idx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["prev5_amt"] = _pmap.reindex(_idx).to_numpy()
    _cum = df.groupby(["instrument", "day"], sort=False)["sum_amt"].cumsum()
    df["c_vshock"] = -(_cum / df["prev5_amt"].where(df["prev5_amt"] > 0))
'''

SNIPS = {"c_conc": SNIP_CONC, "c_maxmin": SNIP_MAXMIN, "c_tilt": SNIP_TILT,
         "c_mdsd": SNIP_MDSD, "c_vshock": SNIP_VSHOCK}


def extras(src, which):
    parts = [SNIPS[w] for w in which]
    return src.replace(ANCHOR, ANCHOR + "".join(parts))


SPECS = {
    "因子代码/123_分钟成交集中度附加.py": (
        ("c_conc",), (0.15,),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 分钟成交集中度附加（新方向，09:30 修复版）。

    新方向的第一个因子：把 30 分钟窗口内部的**分钟级成交分布**拿来做信号。窗口内成交额集中度
      HHI = Σamt²/(Σamt)²（越接近 1 说明成交被少数几分钟吃掉，即大单冲击/流动性冲击），
      取最近 40 个窗口的均值并取负：冲击越大 → 过度反应越强 → 后续反转越明显。
    这是一个此前所有因子都没有用过的信息维度（历史因子只用窗口末价、窗口 VWAP、窗口振幅、分钟涨跌计数）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/124_分钟成交后倾度附加.py": (
        ("c_tilt",), (0.15,),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 窗口内成交时点重心附加（新方向，09:30 修复版）。

    用成交额加权的分钟时点重心（Σamt·t/Σamt）度量「窗口内买盘是在前段还是后段发力」，
      重心越靠窗口后段说明是尾段拉抬，取负。取最近 40 个窗口的均值。
    与时点选择相关的行为金融证据：日内尾段拉升多为情绪性买盘（Barber & Odean 2008 注意力效应），
      次时段容易被回补。09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/125_相对放量附加.py": (
        ("c_vshock",), (0.15,),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 相对放量附加（新方向，09:30 修复版）。

    用「当日累计成交额 ÷ 过去 5 个交易日全天成交额均值」度量个股相对自身历史的放量倍数（只用过去数据，
      无未来信息），取负：相对放量对应情绪性交易与注意力驱动买盘，随后走弱（Gervais 等 2001 成交量与
      后续收益、Barber & Odean 2008）。这是一个跨日、跨窗口的成交结构信号，历史因子未使用。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/126_分钟大单冲击占比附加.py": (
        ("c_maxmin",), (0.15,),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 分钟大单冲击占比附加（新方向，09:30 修复版）。

    单分钟最大成交额 ÷ 窗口成交额（窗口内是否存在一根异常巨量分钟），取最近 40 个窗口的均值并取负。
      与 123 号的 HHI 是同一维度的两种刻画（HHI 是分布层面的集中度，本项是极值层面）；
      两者分别评测，看哪种更能识别「大单冲击之后的过度反应」。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/127_分钟下行波动占比附加.py": (
        ("c_mdsd",), (0.15,),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 1 分钟下行波动占比附加（新方向，09:30 修复版）。

    窗口内「负的 1 分钟收益平方和 ÷ 全部 1 分钟收益平方和」，即 1 分钟粒度的下行波动占比，取负。
      113 号用过 30 分钟窗口粒度的同构指标（有效），本因子把它下推到 1 分钟粒度：粒度更细时
      更能区分「连续被砸」与「一次跳空」，属于对已有效方向换粒度的独立检验。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/128_软同伴两百只三次方加权.py": (
        (), (),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化（同伴取 200 只、相关度 3 次方加权，09:30 修复版）。

    同伴定义的强度外推：114 号用 100 只 / 2 次方（代理分最高之一），本因子用 200 只 / 3 次方，
      检验「更软」是否继续变好，还是已经过拐点。属于同伴维度的一次边界探索，不是新方向本身。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
}

if __name__ == "__main__":
    for path, (which, weights, text) in SPECS.items():
        s = soft_peer(BASE, power=2.0)
        if which:
            s = add_minute_sql(s)
            s = extras(s, which)
            s = keep(s, which)
            s = tail(s, which, weights)
        else:
            s = tail(s, (), ())
            s = s.replace("PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 100, 60, 30, 2.0",
                          "PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 60, 30, 3.0")
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
