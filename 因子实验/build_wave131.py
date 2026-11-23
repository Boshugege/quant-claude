# -*- coding: utf-8 -*-
"""第一百三十一批（把条件反转的分组维度扩到"全维"）：
    756 = 全维条件反转纯源（9 风格 + 真实换手 + 振幅 + 偏度 + 笔数 = 13 维平均；取证版）
    757 = 合法头部（688 配方）但条件反转分量换成全维版
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
HEAD = (CODE / "688_头部加span5同伴045与多风格条件反转020.py").read_text(encoding="utf-8")

OLD5 = '    for _col670 in ["s_BETA", "s_SIZE", "s_LIQUIDTY", "s_RESVOL", "s_MOMENTUM"]:'
NEW9 = '    for _col670 in ["s_BETA", "s_SIZE", "s_LIQUIDTY", "s_RESVOL", "s_MOMENTUM", "s_EARNYILD", "s_GROWTH", "s_LEVERAGE", "s_BTOP"]:'
assert OLD5 in HEAD and NEW9 not in HEAD
S = HEAD.replace(OLD5, NEW9, 1)

ANCHOR = '    df["c_condrev"] = pd.concat(_parts670, axis=1).mean(axis=1)'
assert ANCHOR in S
EXTRA = '''    _turnF = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"].where(df["mcap"] > 0)
    _ampF = roll((df["win_high"] - df["win_low"]) / df["last_px"].where(df["last_px"] > 0), 40, "mean", 10)
    _dealsF = roll(np.log1p(df["sum_deals"]), 40, "mean", 10)
    _dlF = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first")).reset_index()
    _dlF["_dr"] = np.where((_dlF["_last"] > 0) & (_dlF["_pc"] > 0), np.log(_dlF["_last"] / _dlF["_pc"]), np.nan)
    _dlF = _dlF.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dlF["_skew"] = _dlF.groupby("instrument", sort=False)["_dr"].transform(lambda x: x.shift(1).rolling(20, min_periods=12).skew())
    _kF = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _skewF = _dlF.set_index(["instrument", "day"])["_skew"].reindex(_kF).to_numpy()
    for _extF in [_turnF, _ampF, pd.Series(_skewF, index=df.index), _dealsF]:
        _qF = ((_extF.groupby(df["date"]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
        _parts670.append(_rk66 - _rk66.groupby([df["date"], _qF]).transform("mean"))
'''
S = S.replace(ANCHOR, EXTRA + ANCHOR, 1)

# 756：纯源
OLD_SIG = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert OLD_SIG in S
S756 = S.replace(OLD_SIG, '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]''', 1)
S756 = S756.replace("合法头部 + span5 平滑同伴 0.45 + 多风格条件反转平滑 0.20（暴露表错一天）。",
                    "全维条件反转纯源（9 风格 + 真实换手 + 振幅 + 偏度 + 笔数 = 13 维五分位内去均值，无头部叠加、无中性化）；暴露表错一天。", 1)
(CODE / "756_全维条件反转纯源.py").write_text(S756, encoding="utf-8"); print("写出 756")

# 757：头部 + 全维条件反转分量（替换原 c_conds670）
OLD_CONDS = '''    result["c_conds670"] = _ord670.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)'''
assert OLD_CONDS in S
S757 = S.replace("合法头部 + span5 平滑同伴 0.45 + 多风格条件反转平滑 0.20（暴露表错一天）。",
                 "合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。", 1)
(CODE / "757_头部全维条件反转.py").write_text(S757, encoding="utf-8"); print("写出 757")
