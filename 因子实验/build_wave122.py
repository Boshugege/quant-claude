# -*- coding: utf-8 -*-
"""第一百二十二批：隔夜跳空 / 日内收益（开盘→收盘）/ 尾盘 30 分钟收益 的日频分解轴（全部滞后一日及以上）。
    725 = 隔夜跳空 5 日累计（动量方向，待验证）
    726 = 日内收益 5 日累计反转
    727 = 尾盘 30 分钟收益 5 日累计反转
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 725-727：隔夜跳空 / 日内（开盘→收盘）/ 尾盘 30 分钟收益的日频分解（全部滞后一日及以上） =====
    _seg = df["date"].dt.strftime("%H%M")
    df["_px_open"] = df["last_px"].where(_seg == "0930")
    df["_px_mid"] = df["last_px"].where(_seg == "1430")
    df["_px_close"] = df["last_px"].where(_seg == "1500")
    _dly = df.groupby(["day", "instrument"], sort=False).agg(
        _o=("_px_open", "max"), _m=("_px_mid", "max"), _c=("_px_close", "max"),
        _pc=("pre_close", "max"), _amt=("sum_amt", "sum")).reset_index()
    _dly["gap"] = np.where((_dly["_o"] > 0) & (_dly["_pc"] > 0), np.log(_dly["_o"] / _dly["_pc"]), np.nan)
    _dly["intra"] = np.where((_dly["_c"] > 0) & (_dly["_o"] > 0), np.log(_dly["_c"] / _dly["_o"]), np.nan)
    _dly["late"] = np.where((_dly["_c"] > 0) & (_dly["_m"] > 0), np.log(_dly["_c"] / _dly["_m"]), np.nan)
    _dly = _dly.sort_values(["instrument", "day"]).reset_index(drop=True)
    _g = _dly.groupby("instrument", sort=False)
    for _c in ["gap", "intra", "late"]:
        _dly["_" + _c + "_r5"] = _g[_c].transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())
    _kk = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _mm = _dly.set_index(["instrument", "day"])
    df["c_gap5"] = _mm["_gap_r5"].reindex(_kk).to_numpy()
    df["c_intra5"] = -_mm["_intra_r5"].reindex(_kk).to_numpy()
    df["c_late5"] = -_mm["_late_r5"].reindex(_kk).to_numpy()

'''
KEEP_NEW = KEEP_OLD[:-1] + chr(44) + ' "c_gap5", "c_intra5", "c_late5"]'

base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("725_隔夜跳空五日累计", "c_gap5",
     "隔夜跳空五日累计（开盘/前收 的对数比，滞后一日及以上；动量方向，全部暴露表列错一天）"),
    ("726_日内收益五日反转", "c_intra5",
     "日内收益五日累计反转（开盘→收盘 对数比取负、滞后一日及以上；全部暴露表列错一天）"),
    ("727_尾盘收益五日反转", "c_late5",
     "尾盘30分钟收益五日累计反转（14:30→15:00 对数比取负、滞后一日及以上；全部暴露表列错一天）"),
]:
    S = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S = S.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S, encoding="utf-8")
    print("写出", name)
