# -*- coding: utf-8 -*-
"""第八十五批：低通配方推广到盘口主轴 + 一个全新的成交稀疏度轴（零成交窗口占比）。"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

SMOOTH_STAND = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["signal"] = result["c_condrev"]
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, %s)'''
OLD_STAND = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''

# 553：把 485（一至三档挂单不平衡反向，单独版 IC_IR 1.418）按「低通 + 中性化 2.0」重做
B485 = (CODE / "485_单独一至三档挂单不平衡反向.py").read_text(encoding="utf-8")
ASM485 = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ob"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert ASM485 in B485
NEW485 = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ob"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, 2.0)'''
b553 = B485.replace(ASM485, NEW485)
b553 = b553.replace('"""长窗口软同伴', '"""一至三档挂单不平衡反向（485）+ 日内低通 0.6 + 中性化 2.0（低换手版）。\n\n    长窗口软同伴', 1)
(CODE / "553_单独挂单不平衡低通二点零.py").write_text(b553, encoding="utf-8")
print("写出 553_单独挂单不平衡低通二点零.py")

# 554：零成交窗口占比（sum_deals == 0 的 30 分钟窗口比例）当分组 —— 成交稀疏度是没试过的轴
B = B485
ANCHOR = '    df["c_max"] = -roll(win_ret, 40, "max", 10)'
KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_ob"]'
ASM_NEW = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
assert ANCHOR in B and KEEP_OLD in B
RK = '    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5'
COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'
Q = '    _q66 = ((df.groupby("date")["%s"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'
SPARSE = '''    df["_zero"] = (df["sum_deals"] <= 0).astype(float)
    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_z=("_zero", "mean")).reset_index()
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_z"] = _dd66.groupby("instrument", sort=False)["_z"].shift(1)
    _k66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["_sparse"] = pd.Series(_dd66.set_index(["instrument", "day"])["_z"].reindex(_k66).to_numpy(), index=df.index)
''' + RK + "\n" + (Q % "_sparse") + "\n" + COND
b554 = B.replace(ANCHOR, ANCHOR + "\n" + SPARSE).replace(KEEP_OLD, KEEP_OLD[:-1] + ', "c_ob", "c_condrev"]').replace(ASM485, ASM_NEW)
b554 = b554.replace('"""长窗口软同伴', '"""零成交窗口占比（成交稀疏度）分组的条件反转。\n\n    长窗口软同伴', 1)
(CODE / "554_单独零成交窗口占比分组条件反转.py").write_text(b554, encoding="utf-8")
print("写出 554_单独零成交窗口占比分组条件反转.py")

# 555：544 的中性化 2.0 → 2.5；556：544 的低通 0.7 → 0.65
h544 = (CODE / "544_头部加挂单减权低通零点七.py").read_text(encoding="utf-8")
assert "alpha=0.7" in h544 and 'neutralize("signal", CHARS4, 2.0)' in h544
b555 = h544.replace('neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 2.5)')
b555 = b555.replace('"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.7（低通档内细调）。',
                    '"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.5 倍 + 日内低通 0.7（探针）。', 1)
(CODE / "555_头部加挂单零点一低通零点七二点五.py").write_text(b555, encoding="utf-8")
print("写出 555_头部加挂单零点一低通零点七二点五.py")
b556 = h544.replace("alpha=0.7", "alpha=0.65")
b556 = b556.replace('"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.7（低通档内细调）。',
                    '"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.65。', 1)
(CODE / "556_头部加挂单零点一低通零点六五.py").write_text(b556, encoding="utf-8")
print("写出 556_头部加挂单零点一低通零点六五.py")
