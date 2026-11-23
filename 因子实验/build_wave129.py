# -*- coding: utf-8 -*-
"""第一百二十九批（同伴定义 + 更深分组的条件反转；生成后需再过 slim_apply 瘦身）：
    750 = 反转排名在 (BETA五分位 × SIZENL五分位) 25 格内去均值
    751 = 行业内「真实换手率五分位」同伴残差反转（r 减同层同伴均值）
    752 = 行业内「日收益偏度五分位」同伴残差反转
    753 = 反转排名在「隔夜跳空五分位」内去均值
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

OLD_COLS = '_exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]'
NEW_COLS = '_exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE", "ret"]'
BASE = BASE.replace(OLD_COLS, NEW_COLS, 1)

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in BASE

BLOCK = '''    # ===== 新轴 750-753 =====
    _rk9 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _turn9 = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"].where(df["mcap"] > 0)
    _qb = pd.qcut(df["s_BETA"].groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    _qs = pd.qcut(df["s_SIZENL"].groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    _q2d = _qb * 5 + _qs
    df["c_cond2d"] = _rk9 - _rk9.groupby([df["date"], _q2d]).transform("mean")
    _dl9 = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first")).reset_index()
    _dl9["_dr"] = np.where((_dl9["_last"] > 0) & (_dl9["_pc"] > 0), np.log(_dl9["_last"] / _dl9["_pc"]), np.nan)
    _dl9["_gap"] = _dl9["_dr"]
    _dl9 = _dl9.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dl9["_skew9"] = _dl9.groupby("instrument", sort=False)["_dr"].transform(lambda x: x.shift(1).rolling(20, min_periods=12).skew())
    _dl9["_gap9"] = _dl9.groupby("instrument", sort=False)["_gap"].shift(1)
    _kk9 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _mm9 = _dl9.set_index(["instrument", "day"])
    _skew9 = _mm9["_skew9"].reindex(_kk9).to_numpy()
    _gap9 = _mm9["_gap9"].reindex(_kk9).to_numpy()
    _qt = pd.qcut(_turn9.groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    _qk = pd.qcut(pd.Series(_skew9, index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    _qg = pd.qcut(pd.Series(_gap9, index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    df["c_condgap"] = _rk9 - _rk9.groupby([df["date"], _qg]).transform("mean")
    for _tag, _q in [("turnpeer", _qt), ("skewpeer", _qk)]:
        _res9 = []
        for _c in ["r_30", "r_rv4", "r_rv8", "r_day"]:
            _v = df[_c]
            _pm = _v.groupby([df["date"], df["ind_code"], _q]).transform("mean")
            _res9.append(_v - _pm)
        _rkb9 = pd.concat([x.groupby(df["date"]).rank(pct=True) for x in _res9], axis=1).mean(axis=1) - 0.5
        df["c_" + _tag] = -_rkb9

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_cond2d", "c_condgap", "c_turnpeer", "c_skewpeer"]'
base2 = BASE.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("750_两维风格格内条件反转", "c_cond2d",
     "反转排名在 (BETA五分位 × SIZENL五分位) 25 格内去均值；全部暴露表列错一天）"),
    ("751_行业内换手相近同伴残差反转", "c_turnpeer",
     "行业内真实换手率五分位同层同伴残差反转（取负；同伴均值含自身，大层近似）；全部暴露表列错一天）"),
    ("752_行业内偏度相近同伴残差反转", "c_skewpeer",
     "行业内日收益偏度五分位同层同伴残差反转（取负）；全部暴露表列错一天）"),
    ("753_隔夜跳空分组条件反转", "c_condgap",
     "反转排名在前一日隔夜跳空五分位内去均值；全部暴露表列错一天）"),
]:
    S2 = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)
