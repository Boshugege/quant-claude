# -*- coding: utf-8 -*-
"""第七十六批：把两个已验证杠杆叠起来（盘口 1–3 档 × 中性化 2.0 倍、盘口 × 日内低通），
并把盘口不平衡换成"分组变量"的两种新用法（日间变化 / 绝对水平）。

- 516：486（头部 + 一至三档挂单不平衡 0.15，配对 ic_mean_z +7.52 / turn_z −27.32）的 **2.0 倍中性化**版。
- 519：486 + 日内信号低通平滑（EWMA），换手杠杆。
- 517/518：把挂单不平衡当**分组变量**（去均值族骨架），日间变化 / 绝对水平 —— 与 485 把它当信号是两种用法。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
H = (CODE / "486_头部加一至三档挂单不平衡.py").read_text(encoding="utf-8")

# 486 的最终合成块（逐字复制，替换时用作锚点）
ASM = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ob"]
    _rk = (result.groupby("date")[_cols].rank(pct=True) - 0.5).to_numpy()
    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15]), (len(result), 1))
    _late = result["date"].dt.strftime("%H%M").isin(["1400", "1430"]).to_numpy()
    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10])
    _has = ~np.isnan(_rk)
    _den = (_has * _W).sum(axis=1)
    _num = np.nansum(np.nan_to_num(_rk) * _W, axis=1)
    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert ASM in H

SMOOTH = '''    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''

ANCHOR_C = '    df["c_max"] = -roll(win_ret, 40, "max", 10)'
assert ANCHOR_C in H

# 把「挂单不平衡」按股票逐日滞后一格（前一日及更早），再做日间差分 / 取绝对水平。
LAG_BLOCK = '''    # ===== 新增轴：盘口挂单不平衡（1–3 档）的分组用法 =====
    df["_ob"] = _ob  # 486 里 _ob 是局部变量，聚合前先落到列上
    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_o=("_ob", "mean")).reset_index()
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_o"] = _dd66.groupby("instrument", sort=False)["_o"].shift(1)
    _dd66["_do"] = _dd66.groupby("instrument", sort=False)["_o"].diff()
    _k66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _mm66 = _dd66.set_index(["instrument", "day"])
    df["_ob_lag"] = pd.Series(_mm66["_o"].reindex(_k66).to_numpy(), index=df.index)
    df["_ob_dlt"] = pd.Series(_mm66["_do"].reindex(_k66).to_numpy(), index=df.index)'''

COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'
RK = '    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5'
Q = '    _q66 = ((df.groupby("date")["%s"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_ob"]'
assert KEEP_OLD in H
KEEP_NEW = KEEP_OLD[:-1] + ', "c_ob", "c_condrev"]'

SPECS = {
    "516": ("516_头部加挂单不平衡双重杠杆二点零中性化.py", "head2", None),
    "519": ("519_头部加挂单不平衡与日内低通.py", "smooth", None),
    "517": ("517_单独挂单不平衡日间变化分组条件反转.py", "cond", "dlt"),
    "518": ("518_单独挂单不平衡水平分组条件反转.py", "cond", "lvl"),
}

for num, (fname, kind, tag) in SPECS.items():
    if kind == "head2":
        body = H.replace('neutralize("signal", CHARS4, 1.5)', 'neutralize("signal", CHARS4, 2.0)')
        desc = "头部 + 一至三档挂单不平衡 0.15 + 中性化 2.0 倍（把两个已验证杠杆叠起来）"
    elif kind == "smooth":
        anchor2 = ASM.splitlines()[-2:]
        anchor2 = "\n".join(anchor2)
        assert anchor2 in H, "找不到合成两行"
        body = H.replace(anchor2, SMOOTH)
        desc = "头部 + 一至三档挂单不平衡 0.15 + 日内信号低通平滑（换手杠杆）"
    else:
        gcol = "_ob_dlt" if tag == "dlt" else "_ob_lag"
        if tag == "lvl":
            insert = LAG_BLOCK + "\n" + RK + "\n" + '    df["_ob_abs"] = df["_ob_lag"].abs()\n' + Q % "_ob_abs" + "\n" + COND
        else:
            insert = LAG_BLOCK + "\n" + RK + "\n" + Q % gcol + "\n" + COND
        body = H.replace(ANCHOR_C, ANCHOR_C + "\n" + insert).replace(KEEP_OLD, KEEP_NEW)
        body = body.replace(ASM, '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]''')
        desc = ("盘口挂单不平衡日间变化分组的条件反转" if tag == "dlt" else "盘口挂单不平衡水平分组的条件反转")
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
