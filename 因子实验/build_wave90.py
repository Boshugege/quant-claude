# -*- coding: utf-8 -*-
"""第九十批：暴露表信息化的两个明确方向（前一日特质收益反转、风格因子收益动量）
+ 分钟成交额三阶矩（成交集中偏斜）。全部无低通、独立取证。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")

# 1) 暴露表查询里补上 ret（前一日收益），供 577/578 做截面回归
EXP_OLD = '''    _exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]'''
assert EXP_OLD in S561
EXP_NEW = '''    _exp_cols = ["float_market_cap", "industry_level1_code", "ret", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]'''
MAP_OLD = '''    _ind = _exp["industry_level1_code"].reindex(_midx).astype(str)
    df["ind_code"] = _ind.to_numpy()'''
assert MAP_OLD in S561
MAP_NEW = MAP_OLD + '''
    df["e_ret"] = pd.Series(_exp["ret"].reindex(_midx).to_numpy(), index=df.index)'''

STY = '["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]'
REGR = '''
    # ===== 共用：前一日收益对「10 风格暴露 + 申万行业哑变量」做逐日截面回归（全部用错一天的值） =====
    _sty = ["s_" + _c for _c in %s]
    _pf = df.groupby(["day", "instrument"], sort=False).agg(
        _y=("e_ret", "first"), _ind=("ind_code", "first"),
        **{_c: (_c, "first") for _c in _sty}).reset_index()
    _pf = _pf.sort_values(["day", "instrument"]).reset_index(drop=True)
    _res = np.full(len(_pf), np.nan)
    _beta = np.zeros((len(_pf), len(_sty)))
    for _d, _ix in _pf.groupby("day", sort=False).indices.items():
        _sub = _pf.iloc[_ix]
        _y = _sub["_y"].to_numpy(float); _X = _sub[_sty].to_numpy(float)
        _D = pd.get_dummies(_sub["_ind"]).to_numpy(float)
        _ok = (~np.isnan(_y)) & (~np.isnan(_X).any(axis=1)) & (~np.isnan(_D).any(axis=1))
        if _ok.sum() < 100:
            continue
        _A = np.column_stack([np.ones(int(_ok.sum())), _X[_ok], _D[_ok][:, 1:]])
        _b, *_ = np.linalg.lstsq(_A, _y[_ok], rcond=None)
        _res[_ix[_ok]] = _y[_ok] - _A @ _b
        _beta[_ix[_ok]] = _b[1:1 + len(_sty)]
    _pf["_idio"] = _res
    for _i, _c in enumerate(_sty):
        _pf["_b_" + _c] = _beta[:, _i]
''' % STY

BLOCK_577 = REGR + '''
    # ===== 新轴 577：前一日特质收益的 20 日累计反转（残差 = ret 去掉行业与风格） =====
    _pf = _pf.sort_values(["instrument", "day"]).reset_index(drop=True)
    _pf["_idio20"] = _pf.groupby("instrument", sort=False)["_idio"].transform(
        lambda x: x.rolling(20, min_periods=10).sum())
    _pf["_idio20"] = _pf.groupby("instrument", sort=False)["_idio20"].shift(1)
    _k77 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["c_idio20"] = -pd.Series(_pf.set_index(["instrument", "day"])["_idio20"].reindex(_k77).to_numpy(), index=df.index)'''

BLOCK_578 = REGR + '''
    # ===== 新轴 578：风格因子收益的 5 日动量 × 个股自身暴露（stock-level，不会被行业哑变量吸收） =====
    _pf = _pf.sort_values(["day", "instrument"]).reset_index(drop=True)
    _bcols = ["_b_" + _c for _c in _sty]
    for _c in _bcols:
        _pf[_c] = _pf.groupby("day", sort=False)[_c].transform("mean")
    _b5 = _pf.groupby("day", sort=False)[_bcols].mean()
    _b5 = _b5.rolling(5, min_periods=3).mean().shift(1)
    _k78 = pd.MultiIndex.from_arrays([df["day"].values, df["instrument"].values])
    _zs = pd.concat([(df.groupby("date")[_c].rank(pct=True) - 0.5) for _c in _sty], axis=1)
    _zs.columns = _sty
    _bb = _b5.reindex(df["day"].values)
    _bb.index = df.index
    df["c_stymom"] = (_zs.fillna(0.0) * _bb.fillna(0.0).to_numpy()).sum(axis=1)'''

for num, block, col, fname, desc in [
    ("577", BLOCK_577, "c_idio20", "577_单独特质收益二十日反转.py",
     "前一日特质收益 20 日累计反转（残差来自 ret 对 10 风格暴露与申万行业哑变量的逐日截面回归；暴露表全部错一天）"),
    ("578", BLOCK_578, "c_stymom", "578_单独风格因子收益动量.py",
     "风格因子收益 5 日动量 × 个股自身风格暴露（风格因子收益由前一日 ret 的截面回归得到；暴露表错一天）"),
]:
    src = S561.replace(EXP_OLD, EXP_NEW).replace(MAP_OLD, MAP_NEW)
    src = src.replace('    _rk66 = pd.concat', block + '\n    _rk66 = pd.concat', 1)
    src = src.replace('    keep = ["r_30"', '    keep = ["%s", "r_30"' % col, 1)
    src = src.replace('    result["signal"] = result["c_indpeer"]', '    result["signal"] = result["%s"]' % col, 1)
    src = src.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", desc, 1)
    (CODE / fname).write_text(src, encoding="utf-8")
    print("写出", fname)

# 576：窗口内分钟成交额的三阶矩（成交集中偏斜）——在 572 的 SQL 上加一列
s572 = (CODE / "572_单独分钟成交笔数集中度.py").read_text(encoding="utf-8")
s576 = s572.replace("            SUM(amt_c * amt_c) AS sum_amt2,",
                    "            SUM(amt_c * amt_c) AS sum_amt2,\n            SUM(amt_c * amt_c * amt_c) AS sum_amt3,")
s576 = s576.replace("f.sum_amt, f.sum_amt2,", "f.sum_amt, f.sum_amt2, f.sum_amt3,")
s576 = s576.replace('"sum_amt", "sum_amt2",', '"sum_amt", "sum_amt2", "sum_amt3",')
s576 = s576.replace('''    _deals = df["sum_deals"].where(df["sum_deals"] > 0)''',
                    '''    _a = df["sum_amt"].where(df["sum_amt"] > 0)
    _m1 = _a
    _m2 = df["sum_amt2"] / _a
    _m3 = df["sum_amt3"] / _a
    _skew = (_m3 - 3.0 * _m1 * _m2 + 2.0 * _m1 ** 3) / ((_m2 - _m1 ** 2) ** 1.5).where((_m2 - _m1 ** 2) > 1e-18)
    _deals = df["sum_deals"].where(df["sum_deals"] > 0)''')
s576 = s576.replace('    df["c_dealshhi"] = _rk(_smooth(_dealshhi), -1.0)',
                    '    df["c_amtskew"] = _rk(_smooth(_skew), -1.0)')
s576 = s576.replace('keep = ["c_dealshhi"]', 'keep = ["c_amtskew"]')
s576 = s576.replace('result["signal"] = result["c_dealshhi"]', 'result["signal"] = result["c_amtskew"]')
s576 = s576.replace("窗口内分钟成交笔数集中度（HHI），40 窗口均值取负",
                    "窗口内分钟成交额分布的三阶矩（偏斜），40 窗口均值取负")
(CODE / "576_单独分钟成交额偏斜.py").write_text(s576, encoding="utf-8")
print("写出 576_单独分钟成交额偏斜.py")
