# -*- coding: utf-8 -*-
"""第一百二十一批：前一日特质收益反转（723，日频截面回归残差，全列错一天）+ 5 日累计特质反转（724）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

# 1) 暴露表加载里加入 ret 列（会一起被 shift(1)）
OLD_COLS = '_exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]'
NEW_COLS = '_exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE", "ret"]'
assert OLD_COLS in S561
S = S561.replace(OLD_COLS, NEW_COLS, 1)

# 2) 信号块：日频截面回归（滞后收益 ~ 滞后风格 + 行业哑变量）残差反转
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in S
SIG_NEW = '''    # ===== 新轴 723/724：前一日（及更早）特质收益反转（日频截面回归残差，全部错一天） =====
    _dl = _exp.reset_index()
    _sty = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
    _dl = _dl.dropna(subset=["ret"])
    _res = np.full(len(_dl), np.nan)
    _grp = _dl.groupby("day", sort=False)
    for _d, _rows in _grp.indices.items():
        _sub = _dl.iloc[_rows]
        _y = _sub["ret"].to_numpy(float)
        _X = _sub[_sty].to_numpy(float)
        _mu = np.nanmean(_X, axis=0); _sd = np.nanstd(_X, axis=0)
        _X = np.nan_to_num((_X - _mu) / np.where(_sd > 0, _sd, 1.0))
        _ind = pd.get_dummies(_sub["industry_level1_code"].astype(str), drop_first=True).to_numpy(float)
        _X = np.column_stack([np.ones(len(_sub)), _X, _ind])
        _ok = np.isfinite(_y)
        if _ok.sum() < _X.shape[1] + 5:
            continue
        _b, *_ = np.linalg.lstsq(_X[_ok], _y[_ok], rcond=None)
        _res[_rows] = _y - _X @ _b
    _dl["_res"] = _res
    _dl = _dl.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dl["_res5"] = _dl.groupby("instrument", sort=False)["_res"].transform(
        lambda x: x.rolling(5, min_periods=3).sum())
    _k723 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _di = _dl.set_index(["instrument", "day"])
    df["c_idiorev"] = -pd.Series(_di["_res"].reindex(_k723).to_numpy(), index=df.index)
    df["c_idiorev5"] = -pd.Series(_di["_res5"].reindex(_k723).to_numpy(), index=df.index)
    result["signal"] = result["c_idiorev"]'''
S723 = S.replace(SIG_OLD, SIG_NEW, 1).replace(
    "行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）",
    "前一日特质收益反转（滞后一日的日收益对 10 风格 + 行业哑变量做截面回归的残差，取负；全部暴露表列错一天）", 1)
# 723 需要 keep 里带上新列
S723 = S723.replace('"c_condrev", "c_turnself", "c_indrev", "c_indpeer"]',
                    '"c_condrev", "c_turnself", "c_indrev", "c_indpeer", "c_idiorev", "c_idiorev5"]', 1)
(CODE / "723_单独特质日收益反转.py").write_text(S723, encoding="utf-8"); print("写出 723")

S724 = S723.replace('    result["signal"] = result["c_idiorev"]', '    result["signal"] = result["c_idiorev5"]', 1)
S724 = S724.replace("前一日特质收益反转（滞后一日的日收益对 10 风格 + 行业哑变量做截面回归的残差，取负；全部暴露表列错一天）",
                    "5 日累计特质收益反转（滞后日频截面回归残差的 5 日累计，取负；全部暴露表列错一天）", 1)
(CODE / "724_单独特质五日收益反转.py").write_text(S724, encoding="utf-8"); print("写出 724")
