# -*- coding: utf-8 -*-
"""第一百零一批：同伴定义再换四个特征维度（605-608）+ 两个不同机制的独立取证（609 行业滞后收益反转、610 30 分钟尺度路径持续性）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
OLD = '''    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
assert OLD in S561
OLD_DOC = "行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）"
assert OLD_DOC in S561

PEERS = [
    ("605_单独行业内规模五分位同伴残差反转.py",
     '''    # ===== 新轴 605：行业内「前一日规模五分位」同伴残差反转（同伴 = 同行业且前一日 SIZE 暴露同五分位，等权） =====
    _sv605 = df["s_SIZE"]
    _q605 = ((_sv605.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _q605.values]''',
     "行业内前一日规模五分位同伴残差反转（同伴 = 申万一级行业 × 前一日 SIZE 暴露五分位，等权；暴露表错一天）"),
    ("606_单独行业内动量五分位同伴残差反转.py",
     '''    # ===== 新轴 606：行业内「前一日动量五分位」同伴残差反转（同伴 = 同行业且前一日 MOMENTUM 暴露同五分位，等权） =====
    _mv606 = df["s_MOMENTUM"]
    _q606 = ((_mv606.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _q606.values]''',
     "行业内前一日动量五分位同伴残差反转（同伴 = 申万一级行业 × 前一日 MOMENTUM 暴露五分位，等权；暴露表错一天）"),
    ("607_单独行业内价格五分位同伴残差反转.py",
     '''    # ===== 新轴 607：行业内「前一日价格水平五分位」同伴残差反转（同伴 = 同行业且前收同五分位，等权） =====
    _pv607 = df["pre_close"].where(df["pre_close"] > 0)
    _q607 = ((_pv607.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _q607.values]''',
     "行业内前一日价格水平五分位同伴残差反转（同伴 = 申万一级行业 × 前收五分位，等权；暴露表错一天）"),
    ("608_单独行业内流动性五分位同伴残差反转.py",
     '''    # ===== 新轴 608：行业内「前一日流动性五分位」同伴残差反转（同伴 = 同行业且前一日 LIQUIDTY 暴露同五分位，等权） =====
    _lv608 = df["s_LIQUIDTY"]
    _q608 = ((_lv608.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _q608.values]''',
     "行业内前一日流动性五分位同伴残差反转（同伴 = 申万一级行业 × 前一日 LIQUIDTY 暴露五分位，等权；暴露表错一天）"),
]
for fname, block, desc in PEERS:
    b = S561.replace(OLD, block, 1).replace(OLD_DOC, desc, 1)
    (CODE / fname).write_text(b, encoding="utf-8"); print("写出", fname)

# ---- 609：换机制，信号 = 行业前一日收益反转（跨行业动量，561 里已有列，换 signal 取证） ----
b = S561.replace('    result["signal"] = result["c_indpeer"]', '    result["signal"] = result["c_indrev"]', 1)
b = b.replace(OLD_DOC, "行业前一日收益反转（用申万一级行业哑变量把滞后一日的个股收益聚合成行业收益，取负；暴露表错一天）", 1)
(CODE / "609_单独行业前一日收益反转.py").write_text(b, encoding="utf-8"); print("写出 609")

# ---- 610：换机制，信号 = 30 分钟尺度路径持续性（相邻 30 分钟窗口收益同号占比，40 窗口均值取负） ----
NEW610 = '''    _sg610 = np.sign(result["r_30"])
    _pp610 = _sg610 * _sg610.groupby(result["instrument"], sort=False).shift(1)
    result["c_wpers"] = -(_pp610 > 0).where(_pp610.notna()).groupby(result["instrument"], sort=False).transform(
        lambda x: x.rolling(40, min_periods=10).mean())
    result["signal"] = result["c_wpers"]'''
b = S561.replace('    result["signal"] = result["c_indpeer"]', NEW610, 1)
b = b.replace(OLD_DOC, "30 分钟尺度路径持续性（相邻 30 分钟窗口收益同号占比，40 窗口均值取负；只用 1 分钟行情，不读暴露表）", 1)
(CODE / "610_单独三十分钟尺度路径持续性.py").write_text(b, encoding="utf-8"); print("写出 610")
