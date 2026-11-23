# -*- coding: utf-8 -*-
"""第三批新方向（135+）：收盘集合竞价（15:00 bar，从未使用的数据维度）、尾盘节奏、分钟流动性弹性。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, soft_peer, keep, tail, doc  # noqa: E402

CLEAN = "vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02"

SQL_SNAP_AFTER = "            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,"
SQL_SNAP_ADD = """
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,"""

SQL_PX_TAIL = """                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,"""
SQL_PX_TAIL_ADD = SQL_PX_TAIL + """
            CASE
                WHEN hms = '150000' AND %s THEN amt ELSE 0
            END AS amt_cls,
            CASE WHEN px_open > 0 AND px_close > 0 THEN POWER(px_close / px_open - 1, 2) ELSE 0 END AS mret2,""" % CLEAN

SQL_WIN_OLD = "            MAX(pre_close) AS pre_close"
SQL_WIN_ADD = """
            SUM(amt_cls) AS win_cls,
            SUM(mret2) AS sum_mret2"""

SQL_SEL_OLD = "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close"
SQL_SEL_NEW = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close,\n"
               "        f.win_cls, f.sum_mret2")
SQL_FLT_OLD = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:'
SQL_FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
               '              "win_cls", "sum_mret2"]:')

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

PREV_DAYS = '''
    def _prev_days(col, k):
        """按 (instrument, day) 汇总后向前取 k 个交易日均值（只用过去数据），对齐回每一行。"""
        g = df.groupby(["instrument", "day"], sort=False)[col].sum().rename("v").reset_index()
        g = g.sort_values(["instrument", "day"])
        g["p"] = g.groupby("instrument", sort=False)["v"].transform(
            lambda x: x.shift(1).rolling(k, min_periods=3).mean())
        m = g.set_index(["instrument", "day"])["p"]
        idx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
        return pd.Series(m.reindex(idx).to_numpy(), index=df.index)

'''

SNIP_CLS = '''
    # 昨日收盘集合竞价（15:00 bar）成交额 / 过去 5 个交易日收盘竞价成交额均值（全部滞后，全天可用），取负。
    # 当月 15:00 bar 对当日 09:30~14:30 的截面属于未来数据，因此整体滞后一天使用。
    _cls_day = df.groupby(["instrument", "day"], sort=False)["win_cls"].sum().rename("v").reset_index()
    _cls_day = _cls_day.sort_values(["instrument", "day"])
    _g = _cls_day.groupby("instrument", sort=False)["v"]
    _cls_day["lag"] = _g.shift(1)
    _cls_day["pm"] = _g.transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    _idx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _cls_lag = pd.Series(_cls_day.set_index(["instrument", "day"])["lag"].reindex(_idx).to_numpy(), index=df.index)
    _cls_pm = pd.Series(_cls_day.set_index(["instrument", "day"])["pm"].reindex(_idx).to_numpy(), index=df.index)
    df["c_cls"] = -(_cls_lag / _cls_pm.where(_cls_pm > 0))
'''

SNIP_CLSLAG = '''
    # 昨日收盘集合竞价成交额 / 过去 5 个交易日全天成交额均值（整体滞后一天，全天可用），取负。
    _cls_today = df.groupby(["instrument", "day"], sort=False)["win_cls"].transform("sum")
    _cls_day = df.groupby(["instrument", "day"], sort=False)["win_cls"].sum().rename("v").reset_index()
    _cls_day = _cls_day.sort_values(["instrument", "day"])
    _cls_day["p"] = _cls_day.groupby("instrument", sort=False)["v"].shift(1)
    _day_prev = _prev_days("sum_amt", 5)
    _idx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _cls_lag = pd.Series(_cls_day.set_index(["instrument", "day"])["p"].reindex(_idx).to_numpy(), index=df.index)
    df["c_clslag"] = -(_cls_lag / _day_prev.where(_day_prev > 0))
'''

SNIP_TAILVOL = '''
    # 尾盘相对放量：当前窗口成交额 / 过去 5 个交易日全天成交额均值，只在 14:30 截面非退化，取负。
    _day_prev = _prev_days("sum_amt", 5)
    df["c_tailvol"] = -(df["sum_amt"] / _day_prev.where(_day_prev > 0))
'''

SNIP_SHARE = '''
    # 窗口成交额占当日累计成交额比重（时点节奏），取负。
    _cum = df.groupby(["instrument", "day"], sort=False)["sum_amt"].cumsum()
    df["c_share"] = -roll(df["sum_amt"] / _cum.where(_cum > 0), 40, "mean", 10)
'''

SNIP_ELAS = '''
    # 分钟流动性弹性：窗口内 1 分钟收益平方和 / 窗口成交额（分钟级 Amihud 冲击弹性），取负。
    _elas = df["sum_mret2"] / (df["sum_amt"] / 1e8).where(df["sum_amt"] > 0)
    df["c_elas"] = -roll(_elas, 40, "mean", 10)
'''

SNIPS = {"c_cls": SNIP_CLS, "c_clslag": SNIP_CLSLAG, "c_tailvol": SNIP_TAILVOL,
         "c_share": SNIP_SHARE, "c_elas": SNIP_ELAS}


def patch_sql(src):
    assert src.count(SQL_SNAP_AFTER) == 1
    src = src.replace(SQL_SNAP_AFTER, SQL_SNAP_AFTER + SQL_SNAP_ADD)
    assert src.count(SQL_PX_TAIL) == 1
    src = src.replace(SQL_PX_TAIL, SQL_PX_TAIL_ADD)
    assert src.count(SQL_WIN_OLD) == 1
    src = src.replace(SQL_WIN_OLD, SQL_WIN_OLD + "," + SQL_WIN_ADD)
    assert src.count(SQL_SEL_OLD) == 1
    src = src.replace(SQL_SEL_OLD, SQL_SEL_NEW)
    assert src.count(SQL_FLT_OLD) == 1
    return src.replace(SQL_FLT_OLD, SQL_FLT_NEW)


def add_snips(src, cols):
    body = PREV_DAYS
    for c in cols:
        body += SNIPS[c]
    return src.replace(ANCHOR, ANCHOR + body)


SPECS = {
    "因子代码/135_收盘竞价额附加.py": (("c_cls",), (0.15,),
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 收盘集合竞价放量附加（新方向，09:30 修复版）。

    第一次使用 **15:00 收盘集合竞价 bar 的成交额**：信号 = -(昨日收盘竞价成交额 / 过去 5 个交易日收盘竞价
      成交额均值)。（15:00 bar 对当日早盘截面属于未来数据，因此整体滞后一天，8 个截面都可用。）A 股 14:57–15:00 是收盘集合竞价，收盘竞价的放量往往对应机构调仓或资金抢筹，
      对次日开盘价有直接影响（收盘竞价与次日收益的关联，见 Bogousslavsky & Muravyev 2023 关于收盘拍卖的研究）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/136_昨日收盘竞价占比附加.py": (("c_clslag",), (0.15,),
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 昨日收盘竞价占比附加（新方向，09:30 修复版）。

    135 号的滞后可用版本：用**昨日**收盘集合竞价成交额 / 过去 5 个交易日全天成交额均值（整体滞后一天，
      因此 8 个截面都有定义，也是严格的过去信息）。方向同样取负：收盘竞价相对该股正常成交规模越大，
      说明上一个收盘时点的冲击越大，后续越容易回落（Bogousslavsky & Muravyev 2023）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/137_尾盘相对放量附加.py": (("c_tailvol",), (0.15,),
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 尾盘相对放量附加（新方向，09:30 修复版）。

    信号 = -(当前窗口成交额 / 过去 5 个交易日全天成交额均值)：把每个截面的窗口成交额放到该股自身正常
      日成交规模上做标准化，等价于「用成交额时钟衡量当前关注度」；在 14:30 截面即尾盘相对放量。
      与 125 号（当日累计放量）互补：125 用累计口径，本项用单窗口口径。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/138_窗口成交占比附加.py": (("c_share",), (0.15,),
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 窗口成交占比附加（新方向，09:30 修复版）。

    信号 = -(该窗口成交额 ÷ 当日累计成交额) 的 40 窗口均值：刻画「成交在日内的相对时点位置」，
      比 124 号（窗口内分钟重心）更粗一级尺度，两者构成「窗口内 / 日内」两个层级的成交时点结构。
      分母只用当日截至当前窗口的累计值，无未来信息。09:30 修复：同 104 号对照版本。"""),
    "因子代码/139_分钟流动性弹性附加.py": (("c_elas",), (0.15,),
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 分钟流动性弹性附加（新方向，09:30 修复版）。

    信号 = -(窗口内 1 分钟收益平方和 ÷ 窗口成交额) 的 40 窗口均值，即分钟粒度的 Amihud 冲击弹性：
      同样的成交额下价格波动越大，说明承接越差（流动性越差、越容易过度反应）。
      104 号的 c_illiq 是「窗口收益 / 成交额」的 40 窗口均值（30 分钟粒度），本项把分子换成窗口内
      1 分钟收益平方和，度量的是**冲击的即时性**而不是窗口净涨跌，属于新粒度、新含义。
    09:30 修复：同 104 号对照版本。"""),
}

if __name__ == "__main__":
    for path, (cols, weights, text) in SPECS.items():
        s = soft_peer(BASE, power=2.0)
        s = patch_sql(s)
        s = add_snips(s, cols)
        s = keep(s, cols)
        s = tail(s, cols, weights)
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
