# -*- coding: utf-8 -*-
"""第一百二十三批（新信息源探索）：
    728 = 上下影线不对称（Σ上影-Σ下影）/（Σ上影+Σ下影）40窗口均值取负
    729 = 尾盘30分钟成交额占比 40日（滞后一日）均值取负
    730 = 真实换手率水平（40窗口累计成交额 / 前一日流通市值）取负
    731 = 剩余四维（EARNYILD/GROWTH/LEVERAGE/BTOP）风格五分位内条件反转（裸动量排名去均值）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 728-731（新信息源探索） =====
    _asym = (df["sum_upsh"] - df["sum_dnsh"]) / (df["sum_upsh"] + df["sum_dnsh"]).where((df["sum_upsh"] + df["sum_dnsh"]) > 0)
    df["c_wickasym"] = -roll(_asym, 40, "mean", 10)
    _seg2 = df["date"].dt.strftime("%H%M")
    df["_amtlate"] = df["sum_amt"].where(_seg2 == "1500")
    _dl3 = df.groupby(["day", "instrument"], sort=False).agg(
        _amt=("sum_amt", "sum"), _late=("_amtlate", "sum"), _mc=("mcap", "first")).reset_index()
    _dl3["_lshare"] = _dl3["_late"] / _dl3["_amt"].where(_dl3["_amt"] > 0)
    _dl3 = _dl3.sort_values(["instrument", "day"]).reset_index(drop=True)
    _g3 = _dl3.groupby("instrument", sort=False)
    _dl3["_lshare40"] = _g3["_lshare"].transform(lambda x: x.shift(1).rolling(40, min_periods=20).mean())
    _k3 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _m3 = _dl3.set_index(["instrument", "day"])
    df["c_lateshare"] = -_m3["_lshare40"].reindex(_k3).to_numpy()
    df["c_realturn"] = -np.log((roll(df["sum_amt"], 40, "sum", 10) / df["mcap"].where(df["mcap"] > 0)).where(lambda x: x > 0))
    _rk3 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _parts = []
    for _s in ["EARNYILD", "GROWTH", "LEVERAGE", "BTOP"]:
        _q3 = pd.qcut(df["s_" + _s].rank(pct=True), 5, labels=False, duplicates="drop")
        _parts.append(_rk3 - _rk3.groupby([df["date"], _q3]).transform("mean"))
    df["c_condrev4"] = pd.concat(_parts, axis=1).mean(axis=1)

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_wickasym", "c_lateshare", "c_realturn", "c_condrev4"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("728_上下影线不对称", "c_wickasym",
     "上下影线不对称（Σ上影-Σ下影）/（Σ上影+Σ下影）40窗口均值取负；全部暴露表列错一天）"),
    ("729_尾盘成交额占比", "c_lateshare",
     "尾盘30分钟成交额占全天比例（滞后一日的40日均值取负）；全部暴露表列错一天）"),
    ("730_真实换手率水平", "c_realturn",
     "真实换手率水平（40窗口累计成交额 / 前一日流通市值）取负；全部暴露表列错一天）"),
    ("731_剩余风格条件反转", "c_condrev4",
     "剩余四维（EARNYILD/GROWTH/LEVERAGE/BTOP）风格五分位内条件反转（裸动量排名去均值）；全部暴露表列错一天）"),
]:
    S = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S = S.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S, encoding="utf-8")
    print("写出", name)
