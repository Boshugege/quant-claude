# -*- coding: utf-8 -*-
"""第九十二批：多同伴定义合成（同一 SQL）、市值/小市值同伴合成、集合竞价成交占比。全部无低通。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
OLD = '''    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
assert OLD in S561

CORE = '''    _res66 = []
    for _c66 in ["r_30", "r_rv4", "r_rv8", "r_day"]:
        _v66 = df[_c66]
        _ok66 = (_v66.notna() & (_mkw > 0)).to_numpy()
        _vk = np.where(_ok66, np.nan_to_num(_v66.to_numpy()), 0.0)
        _wk = np.where(_ok66, _mkw.to_numpy(), 0.0)
        _gsum = pd.Series(_vk * _wk, index=df.index).groupby(_pk66).transform("sum").to_numpy()
        _gw = pd.Series(_wk, index=df.index).groupby(_pk66).transform("sum").to_numpy()
        _self66 = _vk * _wk
        _den66 = _gw - _wk
        _peer66 = np.where(_den66 > 0, (_gsum - _self66) / np.where(_den66 > 0, _den66, 1.0), np.nan)
        _res66.append(pd.Series(_v66.to_numpy() - _peer66, index=df.index))
    _rkb = pd.concat([x.groupby(df["date"]).rank(pct=True) for x in _res66], axis=1).mean(axis=1) - 0.5'''
assert CORE in S561
CORE_IND = "\n".join(("    " + _l if _l.strip() else _l) for _l in CORE.split("\n"))

HEADER582 = '''    # ===== 新轴 582：四种同伴定义的残差反转合成（行业内市值加权 / 小市值加权 / 前一日换手加权 / 行业×BETA三分位等权） =====
    _dly92 = df.groupby(["day", "instrument"], sort=False).agg(
        _m92=("mcap", "first"), _a92=("sum_amt", "sum")).reset_index()
    _dly92["_t92"] = _dly92["_a92"] / _dly92["_m92"].where(_dly92["_m92"] > 0)
    _dly92 = _dly92.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dly92["_t92"] = _dly92.groupby("instrument", sort=False)["_t92"].shift(1)
    _k92 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _turn92 = pd.Series(_dly92.set_index(["instrument", "day"])["_t92"].reindex(_k92).to_numpy(), index=df.index)
    _qb92 = ((df["s_BETA"].groupby([df["date"], df["ind_code"]]).rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    _parts92 = []
    for _mkw, _pk66 in [
        (df["mcap"].fillna(0.0), [df["date"].values, df["ind_code"].values]),
        ((1.0 / df["mcap"].where(df["mcap"] > 0)).fillna(0.0), [df["date"].values, df["ind_code"].values]),
        (_turn92.where(_turn92 > 0).fillna(0.0), [df["date"].values, df["ind_code"].values]),
        (pd.Series(1.0, index=df.index), [df["date"].values, df["ind_code"].values, _qb92.values]),
    ]:
''' + CORE_IND + '''
        _parts92.append(-_rkb)
    df["c_indpeer"] = pd.concat(_parts92, axis=1).mean(axis=1)'''
b = S561.replace(OLD, HEADER582, 1)
b = b.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）",
              "四种同伴定义（行业内市值加权/小市值加权/前一日换手加权/行业×BETA三分位等权）的残差反转合成；暴露表全部错一天", 1)
(CODE / "582_单独多同伴定义合成反转.py").write_text(b, encoding="utf-8"); print("写出 582")

HEADER584 = '''    # ===== 新轴 584：行业内市值加权与小市值加权两种同伴残差反转的 1:1 合成 =====
    _pa84 = []
    for _mkw, _pk66 in [
        (df["mcap"].fillna(0.0), [df["date"].values, df["ind_code"].values]),
        ((1.0 / df["mcap"].where(df["mcap"] > 0)).fillna(0.0), [df["date"].values, df["ind_code"].values]),
    ]:
''' + CORE_IND + '''
        _pa84.append(-_rkb)
    df["c_indpeer"] = pd.concat(_pa84, axis=1).mean(axis=1)'''
b = S561.replace(OLD, HEADER584, 1)
b = b.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）",
              "行业内市值加权与小市值加权两种同伴残差反转的 1:1 合成（暴露表错一天）", 1)
(CODE / "584_单独市值加权与小市值加权同伴合成.py").write_text(b, encoding="utf-8"); print("写出 584")

S563 = (CODE / "563_单独分钟成交集中度.py").read_text(encoding="utf-8")
s = S563.replace("            CASE\n                WHEN vol > 0 AND amt > 0 AND px_low > 0\n                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02\n                THEN amt ELSE 0\n            END AS amt_c,",
                 "            CASE\n                WHEN vol > 0 AND amt > 0 AND px_low > 0\n                     AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02\n                THEN amt ELSE 0\n            END AS amt_c,\n            CASE WHEN hms = '092500' AND vol > 0 THEN amt ELSE 0 END AS auc_c,")
assert "AS auc_c," in s
s = s.replace("            SUM(amt_c * amt_c) AS sum_amt2,", "            SUM(amt_c * amt_c) AS sum_amt2,\n            SUM(auc_c) AS sum_auc,")
s = s.replace("f.sum_amt, f.sum_amt2,", "f.sum_amt, f.sum_amt2, f.sum_auc,")
s = s.replace('"sum_amt", "sum_amt2",', '"sum_amt", "sum_amt2", "sum_auc",')
s = s.replace('    _hhi = df["sum_amt2"] / (_amt * _amt)',
              '    _auc = df["sum_auc"] / df["sum_amt"].where(df["sum_amt"] > 0)\n    _hhi = df["sum_amt2"] / (_amt * _amt)')
s = s.replace('    df["c_hhi"] = _rk(_smooth(_hhi), -1.0)', '    df["c_auc"] = -_rk(_smooth(_auc), 1.0)')
s = s.replace('keep = ["c_hhi"]', 'keep = ["c_auc"]')
s = s.replace('result["signal"] = result["c_hhi"]', 'result["signal"] = result["c_auc"]')
s = s.replace("窗口内分钟成交额集中度（HHI）的 40 窗口均值，取负：分散成交偏反转",
              "集合竞价（09:25）成交额占当日成交额比例的 40 窗口均值，取负")
(CODE / "583_单独集合竞价成交占比.py").write_text(s, encoding="utf-8"); print("写出 583")
