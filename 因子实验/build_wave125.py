# -*- coding: utf-8 -*-
"""第一百二十五批（承接 723 的解法，扩展特质残差轴）：
    736 = 行业内残差反转（仅 32 行业哑变量回归，滞后日收益残差取负）
    737 = 特质收益 20 日累计反转（10 风格 + 行业残差的 20 日累计取负）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
BASE = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

# 暴露表加载里加入 ret（会一起被 shift(1)）
OLD_COLS = '_exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]'
NEW_COLS = '_exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE", "ret"]'
assert OLD_COLS in BASE
S = BASE.replace(OLD_COLS, NEW_COLS, 1)

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev", "c_indpeer"]'
assert KEEP_OLD in S

BLOCK = '''    # ===== 新轴 736/737：滞后日收益对行业（及风格）的截面回归残差反转 =====
    _dl = _exp.reset_index()
    _sty = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
    _dl = _dl.dropna(subset=["ret"])
    _ind_all = pd.get_dummies(_dl["industry_level1_code"].astype(str), drop_first=True).to_numpy(float)
    _ind_idx = {}
    _c = 0
    for _k in sorted(_dl["industry_level1_code"].astype(str).unique())[1:]:
        _ind_idx[_k] = _c; _c += 1
    _res_all = np.full(len(_dl), np.nan)
    _res_ind = np.full(len(_dl), np.nan)
    _grp = _dl.groupby("day", sort=False)
    for _d, _rows in _grp.indices.items():
        _sub = _dl.iloc[_rows]
        _y = _sub["ret"].to_numpy(float)
        _Xi = _ind_all[_rows]
        _Xs = _sub[_sty].to_numpy(float)
        _mu = np.nanmean(_Xs, axis=0); _sd = np.nanstd(_Xs, axis=0)
        _Xs = np.nan_to_num((_Xs - _mu) / np.where(_sd > 0, _sd, 1.0))
        _ok = np.isfinite(_y)
        _A = np.column_stack([np.ones(len(_sub)), _Xi])
        if _ok.sum() >= _A.shape[1] + 5:
            _b, *_ = np.linalg.lstsq(_A[_ok], _y[_ok], rcond=None)
            _res_ind[_rows] = _y - _A @ _b
        _B = np.column_stack([np.ones(len(_sub)), _Xi, _Xs])
        if _ok.sum() >= _B.shape[1] + 5:
            _b2, *_ = np.linalg.lstsq(_B[_ok], _y[_ok], rcond=None)
            _res_all[_rows] = _y - _B @ _b2
    _dl["_res_ind"] = _res_ind
    _dl["_res_all"] = _res_all
    _dl = _dl.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dl["_res_all20"] = _dl.groupby("instrument", sort=False)["_res_all"].transform(
        lambda x: x.rolling(20, min_periods=10).sum())
    _kk = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _mm = _dl.set_index(["instrument", "day"])
    df["c_indres"] = -_mm["_res_ind"].reindex(_kk).to_numpy()
    df["c_idio20"] = -_mm["_res_all20"].reindex(_kk).to_numpy()

'''
KEEP_NEW = KEEP_OLD[:-1] + ', "c_indres", "c_idio20"]'
base2 = S.replace(KEEP_OLD, BLOCK + KEEP_NEW, 1)
SIG_OLD = '    result["signal"] = result["c_indpeer"]'
assert SIG_OLD in base2

for name, col, doc in [
    ("736_行业残差反转", "c_indres",
     "行业内残差反转（仅 32 行业哑变量回归、滞后日收益残差取负；暴露表全列错一天）"),
    ("737_特质收益二十日反转", "c_idio20",
     "特质收益 20 日累计反转（10 风格+行业残差 20 日累计取负；暴露表全列错一天）"),
]:
    S2 = base2.replace(SIG_OLD, '    result["signal"] = result["%s"]' % col, 1)
    S2 = S2.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", doc, 1)
    (CODE / (name + ".py")).write_text(S2, encoding="utf-8")
    print("写出", name)
