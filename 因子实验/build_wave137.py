# -*- coding: utf-8 -*-
"""第一百三十七批（把"中性化力度"这条本地看不见的杠杆在最好的几个头部配方上补齐）：
    777 = 703 配方（span5同伴040+条件反转020）+ 2.5 倍中性化
    778 = 690 配方（span5同伴045+条件反转010，IC 均值最高）+ 2.0 倍中性化
    779 = 771 配方（头部+原始合成分量 0.25）+ 2.0 倍中性化
    780 = 大合成纯源：同伴残差 + 全维条件反转 + 隔夜跳空分组条件反转 三者等权
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

for src, frac, out, tag in [
    ("703_头部加span5同伴040与多风格条件反转020.py", "2.5", "777_头部换手角二点五中性化.py", "703 配方 + 2.5 倍中性化探针"),
    ("690_头部加span5同伴045与多风格条件反转010.py", "2.0", "778_头部高IC配方二点零中性化.py", "690 配方（IC 均值最高）+ 2.0 倍中性化"),
]:
    s = (CODE / src).read_text(encoding="utf-8")
    old = '    result["factor"] = neutralize("signal", CHARS4, 1.5)'
    assert old in s, src
    s = s.replace(old, '    result["factor"] = neutralize("signal", CHARS4, %s)' % frac, 1)
    (CODE / out).write_text(s, encoding="utf-8")
    print("写出", out)

S771 = (CODE / "771_头部加原始合成分量025.py").read_text(encoding="utf-8")
old = '    result["factor"] = neutralize("signal", CHARS4, 1.5)'
assert old in S771
S779 = S771.replace(old, '    result["factor"] = neutralize("signal", CHARS4, 2.0)', 1)
(CODE / "779_头部加合成分量二点零中性化.py").write_text(S779, encoding="utf-8"); print("写出 779")

S757 = (CODE / "757_头部全维条件反转.py").read_text(encoding="utf-8")
OLD_SIG = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert OLD_SIG in S757
S780 = S757.replace(OLD_SIG, '''    result["signal"] = rank_blend(["c_indpeer", "c_condrev", "c_condgap"], [1.0, 1.0, 1.0])
    result["factor"] = result["signal"]''', 1)
# 780 需要 c_condgap（隔夜跳空分组条件反转）——从 753 的轴里搬过来
GAP_BLOCK = '''    _dlG = df.groupby(["day", "instrument"], sort=False).agg(
        _lastG=("last_px", "last"), _pcG=("pre_close", "first")).reset_index()
    _dlG["_gapG"] = np.where((_dlG["_lastG"] > 0) & (_dlG["_pcG"] > 0), np.log(_dlG["_lastG"] / _dlG["_pcG"]), np.nan)
    _dlG = _dlG.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dlG["_gapG1"] = _dlG.groupby("instrument", sort=False)["_gapG"].shift(1)
    _kG = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    _gapG = _dlG.set_index(["instrument", "day"])["_gapG1"].reindex(_kG).to_numpy()
    _qG = pd.qcut(pd.Series(_gapG, index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    df["c_condgap"] = _rk66 - _rk66.groupby([df["date"], _qG]).transform("mean")
'''
ANCH = '    df["c_condrev"] = pd.concat(_parts670, axis=1).mean(axis=1)'
assert ANCH in S780
S780 = S780.replace(ANCH, ANCH + "\n" + GAP_BLOCK, 1)
S780 = S780.replace('"c_condrev", "c_turnself", "c_indrev", "c_indpeer"]',
                    '"c_condrev", "c_turnself", "c_indrev", "c_indpeer", "c_condgap"]', 1)
(CODE / "780_同伴残差条件反转隔夜合成纯源.py").write_text(S780, encoding="utf-8"); print("写出 780")
