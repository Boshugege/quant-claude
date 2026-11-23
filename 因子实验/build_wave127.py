# -*- coding: utf-8 -*-
"""第一百二十七批（日内截面结构新构造，全部日内逐点变化）：
    742 = 段内收盘对段内 VWAP 的偏离（买盘压力代理，40 窗口均值取负）
    743 = 段内成交额量比（对上一段，40 窗口均值取负）
    744 = 段内成交笔数量比（对上一段，40 窗口均值取负）
    745 = 段内振幅突发（段内振幅 / 40 窗口均值，取负）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 742-745（日内段级结构） =====
    _vwap7 = df["sum_amt"] / df["sum_vol"].where(df["sum_vol"] > 0)
    _prem7 = (df["last_px"] - _vwap7) / _vwap7.where(_vwap7 > 0)
    df["c_vwapprem"] = -roll(_prem7, 40, "mean", 10)
    _prev_amt = df.groupby("instrument", sort=False)["sum_amt"].shift(1)
    _amt_ratio = df["sum_amt"] / _prev_amt.where(_prev_amt > 0)
    df["c_amtratio"] = -roll(_amt_ratio, 40, "mean", 10)
    _prev_dl = df.groupby("instrument", sort=False)["sum_deals"].shift(1)
    _dl_ratio = df["sum_deals"] / _prev_dl.where(_prev_dl > 0)
    df["c_dlratio"] = -roll(_dl_ratio, 40, "mean", 10)
    _amp7 = (df["win_high"] - df["win_low"]) / df["last_px"].where(df["last_px"] > 0)
    _amp_base = roll(_amp7, 40, "mean", 10)
    df["c_ampburst"] = -(_amp7 / _amp_base.where(_amp_base > 0))

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_vwapprem", "c_amtratio", "c_dlratio", "c_ampburst"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("742_段内VWAP偏离", "c_vwapprem",
     "段内收盘对段内 VWAP 偏离（40 窗口均值取负，买盘压力代理）；全部暴露表列错一天）"),
    ("743_段内成交额量比", "c_amtratio",
     "段内成交额对上一段量比（40 窗口均值取负）；全部暴露表列错一天）"),
    ("744_段内成交笔数量比", "c_dlratio",
     "段内成交笔数对上一段量比（40 窗口均值取负）；全部暴露表列错一天）"),
    ("745_段内振幅突发", "c_ampburst",
     "段内振幅相对 40 窗口均值（取负，振幅突发代理）；全部暴露表列错一天）"),
]:
    S2 = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)
