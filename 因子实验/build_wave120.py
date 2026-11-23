# -*- coding: utf-8 -*-
"""第一百二十批：成交节奏（720）+ 涨停代理（721）+ 跌停代理（722）——都是没碰过的信息轴。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S565 = (CODE / "565_单独分钟价格路径持续性.py").read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)

MARK = '    df["c_ppsame"] = _rk(_smooth(_pp), -1.0)\n    keep = ["c_ppsame"]'
assert MARK in S565

# 720：成交节奏 = 当前 30 分钟窗口成交额 / 当日截至当前累计成交额（只用当日已发生数据，合法）
B720 = '''    df["day"] = df["date"].dt.normalize()
    _cum120 = df.groupby(["instrument", "day"], sort=False)["sum_amt"].cumsum()
    _r120 = df["sum_amt"] / _cum120.where(_cum120 > 0)
    df["c_flowrate"] = _rk(_smooth(_r120), 1.0)
    keep = ["c_flowrate"]'''
t = S565.replace(MARK, B720, 1).replace(
    "窗口内相邻分钟收益同号占比，40 窗口均值取负：路径越持续越偏趋势（09:30 置零版）。",
    "成交节奏：当前 30 分钟窗口成交额占当日截至当前累计成交额的比例（40 窗口均值，只用到当日已发生数据）。", 1)
t = t.replace('    result["signal"] = result["c_ppsame"]', '    result["signal"] = result["c_flowrate"]', 1)
wr("720_单独成交节奏.py", t)

# 721：涨停代理 = 前一日及更早的「日收益≥9.5%」在近 20 个交易日中的占比（取负=涨停后反转）
B721 = '''    df["day"] = df["date"].dt.normalize()
    _d120 = df.groupby(["instrument", "day"], sort=False).agg(
        _l=("last_px", "last"), _p=("pre_close", "first")).reset_index()
    _d120["_ret"] = np.where((_d120["_l"] > 0) & (_d120["_p"] > 0), _d120["_l"] / _d120["_p"] - 1.0, np.nan)
    _d120 = _d120.sort_values(["instrument", "day"]).reset_index(drop=True)
    _up = (_d120["_ret"] >= 0.095).astype(float)
    _up = _up.groupby(_d120["instrument"], sort=False).shift(1)          # 只用前一日及更早
    _d120["_upshare"] = _up.groupby(_d120["instrument"], sort=False).transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    _k120 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["c_upshare"] = -pd.Series(_d120.set_index(["instrument", "day"])["_upshare"].reindex(_k120).to_numpy(), index=df.index)
    keep = ["c_upshare"]'''
t = S565.replace(MARK, B721, 1).replace(
    "窗口内相邻分钟收益同号占比，40 窗口均值取负：路径越持续越偏趋势（09:30 置零版）。",
    "涨停代理：前一日及更早「日收益≥9.5%」在近 20 日的占比，取负（涨停后反转；只用前一日及更早的日收益）。", 1)
t = t.replace('    result["signal"] = result["c_ppsame"]', '    result["signal"] = result["c_upshare"]', 1)
wr("721_单独涨停代理.py", t)

# 722：跌停代理（同 721，阈值 −9.5%，符号取正=跌停后反弹）
t = S565.replace(MARK, B721.replace('_up = (_d120["_ret"] >= 0.095)', '_up = (_d120["_ret"] <= -0.095)')
                 .replace('df["c_upshare"] = -pd.Series', 'df["c_upshare"] = pd.Series'), 1).replace(
    "窗口内相邻分钟收益同号占比，40 窗口均值取负：路径越持续越偏趋势（09:30 置零版）。",
    "跌停代理：前一日及更早「日收益≤−9.5%」在近 20 日的占比，取正（跌停后反弹）。", 1)
t = t.replace('    result["signal"] = result["c_ppsame"]', '    result["signal"] = result["c_upshare"]', 1)
wr("722_单独跌停代理.py", t)
