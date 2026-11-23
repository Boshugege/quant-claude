# -*- coding: utf-8 -*-
"""第七十七批：把盘口 1–3 档的结构信息（近档集中度 / 日内变化 / 行业内部残差）做成新信息源，
再加一个「三个已验证杠杆全叠」的头部变体。

骨架：485（含 1–3 档 bv1..bv3/av1..av3 的 SQL）。新源沿用「条件反转去均值」落地方式（与 517/518 同）。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
B = (CODE / "485_单独一至三档挂单不平衡反向.py").read_text(encoding="utf-8")
B519 = (CODE / "519_头部加挂单不平衡与日内低通.py").read_text(encoding="utf-8")

ANCHOR = '    df["c_max"] = -roll(win_ret, 40, "max", 10)'
assert ANCHOR in B
KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_ob"]'
assert KEEP_OLD in B
KEEP_NEW = KEEP_OLD[:-1] + ', "c_ob", "c_condrev"]'
ASM = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ob"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert ASM in B
ASM_NEW = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''

EXP_BLOCK = pathlib.Path("/tmp/expblock.txt").read_text(encoding="utf-8")
EXP_ANCHOR = '    df["day"] = df["date"].dt.normalize()\n'

RK = '    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5'
COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'
Q = '    _q66 = ((df.groupby("date")["%s"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'

LAG_TAIL = '''    _k66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _mm66 = _dd66.set_index(["instrument", "day"])
    df["%s"] = pd.Series(_mm66["%s"].reindex(_k66).to_numpy(), index=df.index)'''

# 520：近档集中度 =（最优买卖档挂单量）/（1–3 档总量），逐日滞后一格。
CONC = '''    df["_ob"] = _ob
    # ===== 新增轴：近档挂单集中度（最优档挂单量占 1–3 档的比例） =====
    df["_conc"] = (df["sum_bidv1"] + df["sum_askv1"]) / (df["sum_bidv"] + df["sum_askv"]).where((df["sum_bidv"] + df["sum_askv"]) > 0)
    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_c=("_conc", "mean")).reset_index()
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_c"] = _dd66.groupby("instrument", sort=False)["_c"].shift(1)
''' + (LAG_TAIL % ("_conc_lag", "_c")) + "\n" + RK + "\n" + (Q % "_conc_lag") + "\n" + COND

# 521：挂单不平衡的日内变化（后段均值 − 前段均值），逐日滞后一格。
SLOPE = '''    df["_ob"] = _ob
    # ===== 新增轴：盘口挂单不平衡的日内变化（下午段 − 上午段） =====
    _hhmm = df["date"].dt.strftime("%H%M")
    df["_e"] = df["_ob"].where(_hhmm.isin(["1000", "1030", "1100"]))
    df["_l"] = df["_ob"].where(_hhmm.isin(["1400", "1430", "1500"]))
    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_e=("_e", "mean"), _l=("_l", "mean")).reset_index()
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_s"] = _dd66["_l"] - _dd66["_e"]
    _dd66["_s"] = _dd66.groupby("instrument", sort=False)["_s"].shift(1)
''' + (LAG_TAIL % ("_obslope_lag", "_s")) + "\n" + RK + "\n" + (Q % "_obslope_lag") + "\n" + COND

# 522：行业内部盘口不平衡残差（行业等权去均值后取负），逐日滞后一格。
INDRES = '''    df["_ob"] = _ob
    # ===== 新增轴：行业内部盘口不平衡残差（行业内等权去均值后取负） =====
    df["_indobres"] = -(df["_ob"] - df["_ob"].groupby([df["date"], df["ind_code"]]).transform("mean"))
    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_r=("_indobres", "mean")).reset_index()
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_r"] = _dd66.groupby("instrument", sort=False)["_r"].shift(1)
''' + (LAG_TAIL % ("_indobres_lag", "_r")) + "\n" + RK + "\n" + (Q % "_indobres_lag") + "\n" + COND

SPECS = {
    "520": ("520_单独近档挂单集中度分组条件反转.py", CONC, "近档挂单集中度分组的条件反转"),
    "521": ("521_单独盘口不平衡日内变化分组条件反转.py", SLOPE, "盘口挂单不平衡日内变化分组的条件反转"),
    "522": ("522_单独行业内部盘口不平衡残差分组条件反转.py", INDRES, "行业内部盘口不平衡残差分组的条件反转"),
}
for num, (fname, block, desc) in SPECS.items():
    body = B.replace(ANCHOR, ANCHOR + "\n" + block).replace(KEEP_OLD, KEEP_NEW).replace(ASM, ASM_NEW)
    if num == "522":  # 行业内部残差需要暴露表的行业列（485 骨架原本不读暴露表）
        assert EXP_ANCHOR in body
        body = body.replace(EXP_ANCHOR, EXP_ANCHOR + EXP_BLOCK, 1)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)

# 524：头部 + 挂单不平衡 0.15 + 中性化 2.0 + 日内低通（三个已验证杠杆全叠）。
assert 'neutralize("signal", CHARS4, 1.5)' in B519
b524 = B519.replace('neutralize("signal", CHARS4, 1.5)', 'neutralize("signal", CHARS4, 2.0)')
b524 = b524.replace('"""长窗口软同伴', '"""头部 + 一至三档挂单不平衡 0.15 + 中性化 2.0 倍 + 日内信号低通平滑（三个已验证杠杆全叠）。\n\n    长窗口软同伴', 1)
(CODE / "524_头部加挂单不平衡低通二点零中性化.py").write_text(b524, encoding="utf-8")
print("写出 524_头部加挂单不平衡低通二点零中性化.py")
