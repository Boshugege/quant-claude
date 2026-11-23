# -*- coding: utf-8 -*-
"""第七十三/七十四批：把已验证的「条件反转」框架（495/497，单独 IC_IR≈1.25、与 104 真相关≈0.82）
换到新的分组变量上，并产出两个头部叠加版。

分组变量 = 新的信息源（暴露表错位后的真换手率 / 前一日市值 / 行业×市值 / 市值 20 日历史分位 /
特质波动 / 规模风格 / 行业×真换手率）。代码骨架与 497 完全一致，只替换分组那一行，
保证与已验证口径同源、可解释、可复现。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

BASE_STANDALONE = (CODE / "497_单独行业内条件反转.py").read_text(encoding="utf-8")
BASE_HEAD = (CODE / "496_头部加风格条件反转反向.py").read_text(encoding="utf-8")

OLD_IND = '''    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''

OLD_STY = '''    _q66 = ((df.groupby("date")["s_LIQUIDTY"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 466/468 实测符号为负，这里取反后使用'''

# 前一交易日的真实换手率（当日全天成交额 ÷ 前一交易日流通市值），只用到截面时点之前的信息。
TURN_LAG = '''    _dt66 = df.groupby(["day", "instrument"], sort=False).agg(
        _a=("sum_amt", "sum"), _m=("mcap", "first")).reset_index()
    _dt66["_t"] = _dt66["_a"] / _dt66["_m"].where(_dt66["_m"] > 0)
    _dt66 = _dt66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dt66["_t"] = _dt66.groupby("instrument", sort=False)["_t"].shift(1)
    _k66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["_turn_lag"] = pd.Series(_dt66.set_index(["instrument", "day"])["_t"].reindex(_k66).to_numpy(), index=df.index)'''

Q_TERC = '    _q66 = ((df.groupby("date")["%s"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'
COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'
INDX = '    _q66 = df["ind_code"].astype(str) + "|" + _q66.astype(str)'


def cond(block):
    return block + "\n" + COND


SPECS = {
    "502": ("502_单独真实换手率分组条件反转.py", "s", cond(TURN_LAG + "\n" + Q_TERC % "_turn_lag"),
            "前一交易日真实换手率三分位内去均值的反转"),
    "503": ("503_单独市值分组条件反转.py", "s", cond(Q_TERC % "mcap"),
            "前一日流通市值三分位内去均值的反转"),
    "504": ("504_单独行业市值双重条件反转.py", "s", cond(Q_TERC % "mcap" + "\n" + INDX),
            "行业×前一日市值双重分组内去均值的反转"),
    "505": ("505_单独市值历史分位分组条件反转.py", "s", cond('''    _dt66 = df.groupby(["day", "instrument"], sort=False).agg(_m=("mcap", "first")).reset_index()
    _dt66 = _dt66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dt66["_p"] = _dt66.groupby("instrument", sort=False)["_m"].transform(
        lambda x: x.rolling(20, min_periods=5).rank(pct=True))
    _k66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _pct = pd.Series(_dt66.set_index(["instrument", "day"])["_p"].reindex(_k66).to_numpy(), index=df.index)
    _q66 = ((_pct.clip(0.0, 1.0) * 3).clip(upper=2.999).fillna(-1)).astype(int)'''),
            "前一日市值在自身 20 日窗口中的历史分位三分位内去均值的反转"),
    "506": ("506_单独特质波动分组条件反转.py", "s", cond(Q_TERC % "c_ivol"),
            "特质波动三分位内去均值的反转"),
    "507": ("507_单独风格规模分组条件反转.py", "s", cond(Q_TERC % "s_SIZE"),
            "前一日规模风格暴露三分位内去均值的反转"),
    "508": ("508_单独行业真换手率双重条件反转.py", "s", cond(TURN_LAG + "\n" + Q_TERC % "_turn_lag" + "\n" + INDX),
            "行业×前一交易日真换手率双重分组内去均值的反转"),
    "509": ("509_单独低换手加权反转交互.py", "s",
            TURN_LAG + '''
    _w66 = 0.5 - df.groupby("date")["_turn_lag"].rank(pct=True)
    df["c_condrev"] = _rk66 * _w66''',
            "反转 × 前一交易日真换手率交互（低换手加权，结构与去均值族不同）"),
    "510": ("510_头部加市值分组条件反转.py", "h", cond(Q_TERC % "mcap"),
            "头部 + 市值分组条件反转 0.15"),
    "511": ("511_头部加真换手率分组条件反转.py", "h", cond(TURN_LAG + "\n" + Q_TERC % "_turn_lag"),
            "头部 + 真换手率分组条件反转 0.15"),
}

for num, (fname, kind, block, desc) in SPECS.items():
    base = BASE_HEAD if kind == "h" else BASE_STANDALONE
    old = OLD_STY if kind == "h" else OLD_IND
    assert old in base, (num, "骨架里找不到待替换的分组行")
    body = base.replace(old, block)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
