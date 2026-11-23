# -*- coding: utf-8 -*-
"""第一百二十六批（新源探索 + 符号对冲）：
    738 = 日内成交时段重心（成交额的时间重心，滞后一日 40 日均值取负）
    739 = 行业内真实换手率相对位置（行业内 rank 去均值取负）
    740 = 行业内特质波动相对位置（行业内 rank 去均值取负）
    741 = 前一日隔夜跳空反转（1 日，取负；与 725 的动量方向互为对冲）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 738-741 =====
    _sego = {"0930": 0, "1000": 1, "1030": 2, "1100": 3, "1130": 4, "1300": 5, "1330": 6, "1400": 7, "1430": 8, "1500": 9}
    _sid = df["date"].dt.strftime("%H%M").map(_sego).astype(float)
    _dl6 = df.groupby(["day", "instrument"], sort=False).agg(
        _amt=("sum_amt", "sum"), _pc=("pre_close", "max"),
        _hi=("win_high", "max"), _lo=("win_low", "min"),
        _last=("last_px", "last"), _ind=("ind_code", "first")).reset_index()
    _sv = pd.Series(_sid.to_numpy(float), index=df.index)
    _num = (_sv * df["sum_amt"]).groupby([df["day"], df["instrument"]], sort=False).sum()
    _num.index.names = ["day", "instrument"]
    _dl6 = _dl6.merge(_num.rename("_num").reset_index(), on=["day", "instrument"], how="left")
    _dl6["_tcent"] = _dl6["_num"] / _dl6["_amt"].where(_dl6["_amt"] > 0)
    _dl6["_gap"] = np.where((_dl6["_last"] > 0) & (_dl6["_pc"] > 0), np.log(_dl6["_last"] / _dl6["_pc"]), np.nan)
    _dl6 = _dl6.sort_values(["instrument", "day"]).reset_index(drop=True)
    _g6 = _dl6.groupby("instrument", sort=False)
    _dl6["_tcent40"] = _g6["_tcent"].transform(lambda x: x.shift(1).rolling(40, min_periods=20).mean())
    _dl6["_gap1"] = _g6["_gap"].shift(1)
    _k6 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _m6 = _dl6.set_index(["instrument", "day"])
    df["c_tcent"] = -_m6["_tcent40"].reindex(_k6).to_numpy()
    df["c_gaprev"] = -_m6["_gap1"].reindex(_k6).to_numpy()
    _tr6 = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"].where(df["mcap"] > 0)
    _rp6 = _tr6.groupby([df["date"], df["ind_code"]]).rank(pct=True) - 0.5
    df["c_indtr"] = -_rp6
    _ivol6 = roll(idio, 40, "std", 10)
    _rv6 = _ivol6.groupby([df["date"], df["ind_code"]]).rank(pct=True) - 0.5
    df["c_indvol"] = -_rv6

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_tcent", "c_gaprev", "c_indtr", "c_indvol"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("738_成交时段重心", "c_tcent",
     "日内成交额时间重心（滞后一日 40 日均值取负）；全部暴露表列错一天）"),
    ("739_行业内真实换手相对位置", "c_indtr",
     "行业内真实换手率相对位置（成交额/前一日流通市值 的行业内排名去均值取负）；全部暴露表列错一天）"),
    ("740_行业内特质波动相对位置", "c_indvol",
     "行业内特质波动相对位置（特质波动行业内排名去均值取负）；全部暴露表列错一天）"),
    ("741_隔夜跳空一日反转", "c_gaprev",
     "前一日隔夜跳空反转（开盘/前收对数比滞后一日取负）；全部暴露表列错一天）"),
]:
    S2 = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)
