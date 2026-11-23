# -*- coding: utf-8 -*-
"""第一百批：四个超低换手微观轴一次性复合叠加进头部（601）+ 三个新同伴定义（602/603/604，复用 497 SQL）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

# ---- 601：由 579（头部 + LAG 层 + 路径持续性）扩展，补 HHI/偏斜/笔数集中度 ----
s = (CODE / "579_头部加分钟价格路径持续性015.py").read_text(encoding="utf-8")
s = s.replace("            SUM(amt_c) AS sum_amt,",
              "            SUM(amt_c) AS sum_amt,\n            SUM(amt_c * amt_c) AS sum_amt2,\n            SUM(amt_c * amt_c * amt_c) AS sum_amt3,\n            SUM(deals) AS sum_deals,\n            SUM(deals * deals) AS sum_deals2,", 1)
s = s.replace("f.sum_amt, f.n_vpsync,", "f.sum_amt, f.sum_amt2, f.sum_amt3, f.sum_deals, f.sum_deals2, f.n_vpsync,", 1)
s = s.replace('for c in ["sum_amt", "n_vpsync",',
              'for c in ["sum_amt", "sum_amt2", "sum_amt3", "sum_deals", "sum_deals2", "n_vpsync",', 1)
NEWAX = '''    _n100 = df.groupby(["date","instrument"], sort=False)["instrument"].transform("size")
    _a100 = df["sum_amt"].where(df["sum_amt"] > 0)
    _ds100 = df["sum_deals"].where(df["sum_deals"] > 0)
    df["c_dealshhi"] = -(df["sum_deals2"] / (_ds100 * _ds100)).groupby(df["instrument"], sort=False).transform(
        lambda x: x.rolling(40, min_periods=10).mean())
    df["c_amthhi"] = -(df["sum_amt2"] / (_a100 * _a100)).groupby(df["instrument"], sort=False).transform(
        lambda x: x.rolling(40, min_periods=10).mean())
'''
s = s.replace('    _syncbb = df["n_vpsync"]', NEWAX + '    _syncbb = df["n_vpsync"]', 1)
s = s.replace('"c_body", "c_ob", "c_ppsame"]', '"c_body", "c_ob", "c_ppsame", "c_vpsync", "c_dealshhi", "c_amthhi"]', 1)
s = s.replace("np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.30])", "np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.10, 0.10, 0.10, 0.10])")
s = s.replace("np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10, 0.30])", "np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10, 0.08, 0.08, 0.08, 0.08])")
s = s.replace('    keep = ["c_ppsame", "r_30"', '    keep = ["c_ppsame", "c_vpsync", "c_dealshhi", "c_amthhi", "r_30"', 1)
s = s.replace("合法头部 + 分钟价格路径持续性 0.15（超低换手、与 104 近乎无关的新轴）",
              "合法头部 + 四个超低换手微观轴（路径持续性/量价同步/笔数集中度/成交额集中度）各 0.10 的复合叠加")
(CODE / "601_头部加四个微观轴复合叠加.py").write_text(s, encoding="utf-8"); print("写出 601")

# ---- 602/603/604：换同伴定义（复用 497 SQL；先用 561 的负号取证） ----
S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
OLD = '''    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
assert OLD in S561
for fname, block, desc in [
    ("602_单独行业内非流动性相近同伴残差反转.py",
     '''    # ===== 新轴 602：行业内「前一日非流动性相近」同伴残差反转（同伴 = 同行业且前一日非流动性同三分位，等权） =====
    _ill66 = df["c_illiq"]
    _qi66 = ((_ill66.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _qi66.values]''',
     "行业内前一日非流动性相近同伴残差反转（同伴 = 申万一级行业 × 前一日非流动性三分位；暴露表错一天）"),
    ("603_单独行业内特质波动相近同伴残差反转.py",
     '''    # ===== 新轴 603：行业内「前一日特质波动相近」同伴残差反转（同伴 = 同行业且前一日特质波动同三分位，等权） =====
    _iv66 = df["c_ivol"]
    _qv66 = ((_iv66.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _qv66.values]''',
     "行业内前一日特质波动相近同伴残差反转（同伴 = 申万一级行业 × 前一日特质波动三分位；暴露表错一天）"),
    ("604_单独全市场换手加权同伴残差反转.py",
     '''    # ===== 新轴 604：全市场「前一日换手加权」同伴残差反转（权重 = 当日成交额 / 前一日流通市值） =====
    _dly600 = df.groupby(["day", "instrument"], sort=False).agg(
        _m600=("mcap", "first"), _a600=("sum_amt", "sum")).reset_index()
    _dly600["_t600"] = _dly600["_a600"] / _dly600["_m600"].where(_dly600["_m600"] > 0)
    _dly600 = _dly600.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dly600["_t600"] = _dly600.groupby("instrument", sort=False)["_t600"].shift(1)
    _k600 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _turn600 = pd.Series(_dly600.set_index(["instrument", "day"])["_t600"].reindex(_k600).to_numpy(), index=df.index)
    _mkw = _turn600.where(_turn600 > 0).fillna(0.0)
    _pk66 = [df["date"].values]''',
     "全市场前一日换手加权同伴残差反转（不按行业切分；暴露表错一天）"),
]:
    b = S561.replace(OLD, block, 1)
    b = b.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）", desc, 1)
    (CODE / fname).write_text(b, encoding="utf-8"); print("写出", fname)
