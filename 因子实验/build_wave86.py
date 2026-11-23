# -*- coding: utf-8 -*-
"""第八十六批：两个没试过的日内结构轴（隔夜跳空、上午成交占比）当分组 + 低通 alpha 上探 + 稀疏度配方版。"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

S497 = (CODE / "497_单独行业内条件反转.py").read_text(encoding="utf-8")
OLD = '''    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
ANCHOR = '    df["c_max"] = -roll(win_ret, 40, "max", 10)'
KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_condrev", "c_turnself", "c_indrev"]'
for s in (OLD, ANCHOR, KEEP_OLD):
    assert s in S497
RK = '    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5'
COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'
Q = '    _q66 = ((df.groupby("date")["%s"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'
LAGTAIL = '''    _k66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["%s"] = pd.Series(_dd66.set_index(["instrument", "day"])["%s"].reindex(_k66).to_numpy(), index=df.index)'''

# 557：前一日开盘跳空（09:30 段的 −r_30 即 log(开盘/昨收)）分组
GAP = '''    _hh66 = df["date"].dt.strftime("%H%M")
    df["_gap"] = -df["r_30"].where(_hh66 == "0930")
    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_g=("_gap", "mean")).reset_index()
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_g"] = _dd66.groupby("instrument", sort=False)["_g"].shift(1)
''' + (LAGTAIL % ("_gap_lag", "_g")) + "\n" + RK + "\n" + (Q % "_gap_lag") + "\n" + COND

# 558：前一日上午成交额占全天比例分组
AMSH = '''    _am66 = df["date"].dt.strftime("%H%M") <= "1130"
    df["_amt_am"] = df["sum_amt"].where(_am66, 0.0)
    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_a=("_amt_am", "sum"), _t=("sum_amt", "sum")).reset_index()
    _dd66["_r"] = _dd66["_a"] / _dd66["_t"].where(_dd66["_t"] > 0)
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_r"] = _dd66.groupby("instrument", sort=False)["_r"].shift(1)
''' + (LAGTAIL % ("_amsh_lag", "_r")) + "\n" + RK + "\n" + (Q % "_amsh_lag") + "\n" + COND

PLACEHOLDER = '    df["c_condrev"] = np.nan  # 占位：真正的分组在后面插块时重算'
for num, block, fname, desc in [
    ("557", GAP, "557_单独隔夜跳空分组条件反转.py", "前一日开盘跳空分组的条件反转"),
    ("558", AMSH, "558_单独上午成交占比分组条件反转.py", "前一日上午成交额占比分组的条件反转"),
]:
    body = S497.replace(OLD, PLACEHOLDER).replace(ANCHOR, ANCHOR + "\n" + block)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)

# 559：544 的低通 0.7 → 0.8（在挂单 0.10 + 2.5 的档上继续上探）
h555 = (CODE / "555_头部加挂单零点一低通零点七二点五.py").read_text(encoding="utf-8")
assert "alpha=0.7" in h555
b559 = h555.replace("alpha=0.7", "alpha=0.8")
b559 = b559.replace('"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.5 倍 + 日内低通 0.7（探针）。',
                    '"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.5 倍 + 日内低通 0.8。', 1)
(CODE / "559_头部加挂单零点一低通零点八二点五.py").write_text(b559, encoding="utf-8")
print("写出 559_头部加挂单零点一低通零点八二点五.py")

# 560：零成交窗口占比分组的配方版（低通 + 2.0）
SMOOTH = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["signal"] = result["c_condrev"]
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, 2.0)'''
OLD_STAND = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
b560 = (CODE / "554_单独零成交窗口占比分组条件反转.py").read_text(encoding="utf-8")
assert OLD_STAND in b560
b560 = b560.replace(OLD_STAND, SMOOTH)
b560 = b560.replace('"""长窗口软同伴', '"""零成交窗口占比分组 + 低通 0.6 + 中性化 2.0（稀疏度轴的配方版）。\n\n    长窗口软同伴', 1)
(CODE / "560_单独零成交窗口占比分组低通二点零.py").write_text(b560, encoding="utf-8")
print("写出 560_单独零成交窗口占比分组低通二点零.py")
