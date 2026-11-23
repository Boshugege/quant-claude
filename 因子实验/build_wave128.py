# -*- coding: utf-8 -*-
"""第一百二十八批（条件反转族的分组维度扩展——本轮验证过的最有效方向）：
    746 = 真实换手率五分位内条件反转
    747 = 日内振幅五分位内条件反转
    748 = 日收益偏度五分位内条件反转
    749 = 成交笔数五分位内条件反转
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 746-749：反转排名在"非风格"维度五分位内去均值 =====
    _rk8 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _turn8 = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"].where(df["mcap"] > 0)
    _amp8 = roll((df["win_high"] - df["win_low"]) / df["last_px"].where(df["last_px"] > 0), 40, "mean", 10)
    _dl8 = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first")).reset_index()
    _dl8["_dr"] = np.where((_dl8["_last"] > 0) & (_dl8["_pc"] > 0), np.log(_dl8["_last"] / _dl8["_pc"]), np.nan)
    _dl8 = _dl8.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dl8["_skew8"] = _dl8.groupby("instrument", sort=False)["_dr"].transform(lambda x: x.shift(1).rolling(20, min_periods=12).skew())
    _kk8 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _mm8 = _dl8.set_index(["instrument", "day"])
    df["_skew8"] = _mm8["_skew8"].reindex(_kk8).to_numpy()
    _deals8 = roll(np.log1p(df["sum_deals"]), 40, "mean", 10)
    for _tag, _col in [("turn", _turn8), ("amp", _amp8), ("skew", df["_skew8"]), ("deals", _deals8)]:
        _q8 = pd.qcut(pd.Series(_col, index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
        df["c_cond_" + _tag] = _rk8 - _rk8.groupby([df["date"], _q8]).transform("mean")

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_cond_turn", "c_cond_amp", "c_cond_skew", "c_cond_deals"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("746_真实换手五分位条件反转", "c_cond_turn",
     "反转排名在真实换手率五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
    ("747_日内振幅五分位条件反转", "c_cond_amp",
     "反转排名在日内振幅五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
    ("748_日收益偏度五分位条件反转", "c_cond_skew",
     "反转排名在日收益偏度五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
    ("749_成交笔数五分位条件反转", "c_cond_deals",
     "反转排名在成交笔数五分位内去均值（条件反转族新分组维度）；全部暴露表列错一天）"),
]:
    S2 = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)
