# -*- coding: utf-8 -*-
"""第七十五批：日内路径类新信息源（半方差不对称 / 跳跃占比）+ 换手杠杆（日内信号低通平滑）。

- 512/515：把「反转 × 上行/下行波动不对称」当分组变量（新信息源，仍是去均值族骨架，方向沿用已验证正号）。
- 513：把「日内跳跃占比」当分组变量（新信息源）。
- 514：头部 + 日内信号低通平滑（EWMA，换手杠杆，头部变体）。
骨架仍是 497（单独）/496（头部），与 495/497 同源。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S = (CODE / "497_单独行业内条件反转.py").read_text(encoding="utf-8")
H = (CODE / "496_头部加风格条件反转反向.py").read_text(encoding="utf-8")

OLD_H = '''    _q66 = ((df.groupby("date")["s_LIQUIDTY"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 466/468 实测符号为负，这里取反后使用'''

OLD_S = '''    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''

# 上行/下行波动不对称：40 窗口内正收益标准差 − 负收益标准差（>0 表示下行波动更大）。
SEMIVAR = '''    _up66 = ret30.where(ret30 > 0)
    _dn66 = ret30.where(ret30 < 0)
    df["_semiv"] = roll(_dn66, 40, "std", 10) - roll(_up66, 40, "std", 10)
    _q66 = ((df.groupby("date")["_semiv"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'''

# 跳跃占比：|收益| 超过 3 倍滚动波动率的截面数占比（40 窗口）。
JUMP = '''    _sd66 = roll(ret30, 40, "std", 10)
    _jm66 = (ret30.abs() > 3.0 * _sd66).where(ret30.notna() & _sd66.notna()).astype(float)
    df["_jump"] = roll(_jm66, 40, "mean", 10)
    _q66 = ((df.groupby("date")["_jump"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'''

COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'

SPECS = {
    "512": ("512_单独半方差不对称分组条件反转.py", "s", SEMIVAR + "\n" + COND,
            "上行/下行波动不对称三分位内去均值的反转"),
    "513": ("513_单独跳跃占比分组条件反转.py", "s", JUMP + "\n" + COND,
            "日内跳跃占比三分位内去均值的反转"),
    "515": ("515_头部加半方差不对称分组条件反转.py", "h", SEMIVAR + "\n" + COND,
            "头部 + 半方差不对称分组条件反转 0.15"),
    "514": ("514_头部加日内信号低通平滑.py", "h", None,
            "头部 + 日内信号 EWMA 低通平滑（换手杠杆，头部变体）"),
}

for num, (fname, kind, block, desc) in SPECS.items():
    base = H if kind == "h" else S
    if num == "514":
        # 头部骨架：在 neutralize 之前先把 signal 沿「同一股票同一天」做低通平滑，压掉日内高频翻转。
        old = '''    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
        new = '''    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
        assert old in base
        body = base.replace(old, new)
    else:
        eq = OLD_H if kind == "h" else OLD_S
        assert eq in base, num
        body = base.replace(eq, block)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
