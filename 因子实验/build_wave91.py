# -*- coding: utf-8 -*-
"""第九十一批：头部叠加两个超低换手微观新轴（路径持续性、量价同步性）+ 一个换同伴定义的新源。
579/580 由 486（合法头部 + 挂单）改造：往它的 SQL 里注入分钟序列层，pandas 段加新轴权重 0.15。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S486 = (CODE / "486_头部加一至三档挂单不平衡.py").read_text(encoding="utf-8")

# --- 1) 在 cte_window 之前插入分钟序列层 ---
ANCHOR = "    cte_window AS (\n        SELECT\n            trading_day, time_segment, instrument_id,\n            SUM(amt_c) AS sum_amt,\n"
assert ANCHOR in S486
SEQ = """    cte_seq AS (
        SELECT
            *,
            LAG(px) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS px_l1,
            LAG(px, 2) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS px_l2,
            LAG(amt_c) OVER (PARTITION BY instrument_id, trading_day ORDER BY hms) AS amt_l1
        FROM cte_snapshot
    ),
"""
BODY = """    cte_window AS (
        SELECT
            trading_day, time_segment, instrument_id,
            SUM(amt_c) AS sum_amt,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND (px - px_l1) <> 0 AND (amt_c - amt_l1) <> 0 AND (px - px_l1) * (amt_c - amt_l1) > 0 THEN 1 ELSE 0 END) AS n_vpsync,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND (px - px_l1) <> 0 AND (amt_c - amt_l1) <> 0 THEN 1 ELSE 0 END) AS n_vppair,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 AND (px - px_l1) <> 0 AND (px_l1 - px_l2) <> 0 AND (px - px_l1) * (px_l1 - px_l2) > 0 THEN 1 ELSE 0 END) AS n_ppsame,
            SUM(CASE WHEN px > 0 AND px_l1 > 0 AND px_l2 > 0 AND (px - px_l1) <> 0 AND (px_l1 - px_l2) <> 0 THEN 1 ELSE 0 END) AS n_pppair,
"""
s = S486.replace(ANCHOR, SEQ + BODY, 1)
s = s.replace("        FROM cte_snapshot\n        WHERE time_segment != -1", "        FROM cte_seq\n        WHERE time_segment != -1", 1)
# 最终 SELECT 与 astype 列表补列
s = s.replace("f.sum_amt, f.sum_vol,", "f.sum_amt, f.n_vpsync, f.n_vppair, f.n_ppsame, f.n_pppair, f.sum_vol,", 1)
s = s.replace('for c in ["sum_amt", "sum_vol",', 'for c in ["sum_amt", "n_vpsync", "n_vppair", "n_ppsame", "n_pppair", "sum_vol",', 1)
# 新轴特征
JET = '''    c_ob = None
'''
PATCH = '''    _syncbb = df["n_vpsync"] / df["n_vppair"].where(df["n_vppair"] > 0)
    _ppbb = df["n_ppsame"] / df["n_pppair"].where(df["n_pppair"] > 0)
    df["c_vpsync"] = -_syncbb.groupby(df["instrument"], sort=False).transform(
        lambda x: x.rolling(40, min_periods=10).mean())
    df["c_ppsame"] = -_ppbb.groupby(df["instrument"], sort=False).transform(
        lambda x: x.rolling(40, min_periods=10).mean())
'''
s = s.replace('    df["c_ob"] = -_ob', PATCH + '    df["c_ob"] = -_ob', 1)

for fname, axis, w, desc in [
    ("579_头部加分钟价格路径持续性015.py", "c_ppsame", "0.15", "合法头部 + 分钟价格路径持续性 0.15（超低换手、与 104 近乎无关的新轴）"),
    ("580_头部加量价同步性015.py", "c_vpsync", "0.15", "合法头部 + 量价同步性 0.15（超低换手、与 104 近乎无关的新轴）"),
]:
    b = s.replace('"c_body", "c_ob"]', '"c_body", "c_ob", "%s"]' % axis, 1)
    b = b.replace("np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15])",
                  "np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15, %s])" % w)
    b = b.replace("np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10])",
                  "np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10, %s])" % w)
    b = b.replace('    keep = ["r_30"', '    keep = ["%s", "r_30"' % axis, 1)
    b = b.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(b, encoding="utf-8")
    print("写出", fname)

# 581：行业内前一日 BETA 相近同伴残差反转（同伴 = 同行业且前一日 BETA 同三分位）
S561 = (CODE / "561_单独行业内市值加权同伴残差反转.py").read_text(encoding="utf-8")
OLD = '''    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
assert OLD in S561
NEW = '''    _beta66 = df["s_BETA"]
    _qb66 = ((_beta66.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _qb66.values]'''
b581 = S561.replace(OLD, NEW, 1)
b581 = b581.replace("行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）",
                    "行业内前一日 BETA 相近同伴残差反转（同伴 = 申万一级行业 × 前一日 BETA 三分位，等权；暴露表错一天）", 1)
(CODE / "581_单独行业内BETA相近同伴残差反转.py").write_text(b581, encoding="utf-8")
print("写出 581_单独行业内BETA相近同伴残差反转.py")
