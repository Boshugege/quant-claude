# -*- coding: utf-8 -*-
"""第一百三十五批：
    770 = 同伴残差（行业市值加权，正号）+ 全维条件反转（原始）等权排名合成纯源（取证版）
    771 = 头部（757 配方）+ 上面的合成分量 0.25
    772 = 头部（757 配方）+ 3.0 倍中性化（探针）
    773 = 合成源平滑 span5 纯源（低换手版）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S757 = (CODE / "757_头部全维条件反转.py").read_text(encoding="utf-8")

COMBO = '    result["c_rawcombo"] = rank_blend(["c_indpeer", "c_condrev"], [1.0, 1.0])\n'
ORD = '    result["ladder"] = peer_ladder()\n'
assert ORD in S757
S0 = S757.replace(ORD, ORD + COMBO, 1)

OLD_SIG = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert OLD_SIG in S0

S770 = S0.replace(OLD_SIG, '''    result["signal"] = rank_blend(["c_indpeer", "c_condrev"], [1.0, 1.0])
    result["factor"] = result["signal"]''', 1)
S770 = S770.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "同伴残差（行业市值加权）+ 全维条件反转（13 维）等权排名合成纯源（无头部叠加、无中性化）；暴露表错一天。", 1)
(CODE / "770_同伴残差与全维条件反转合成纯源.py").write_text(S770, encoding="utf-8"); print("写出 770")

S771 = S0.replace('''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])''',
'''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670", "c_rawcombo"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20, 0.25])''', 1)
S771 = S771.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "头部（757 配方）+ 同伴残差与全维条件反转的原始合成分量 0.25；暴露表错一天。", 1)
(CODE / "771_头部加原始合成分量025.py").write_text(S771, encoding="utf-8"); print("写出 771")

S772 = S0.replace('    result["factor"] = neutralize("signal", CHARS4, 1.5)',
                  '    result["factor"] = neutralize("signal", CHARS4, 3.0)', 1)
S772 = S772.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "头部（757 配方）+ 3.0 倍中性化探针（力度越大私榜风险越高，仅作探针）；暴露表错一天。", 1)
(CODE / "772_头部全维三点零中性化探针.py").write_text(S772, encoding="utf-8"); print("写出 772")

S773 = S0.replace('''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])
    result["factor"] = neutralize("signal", CHARS4, 1.5)''',
'''    _o773 = result.sort_values(["instrument", "date"])
    result["c_rawcombo_s"] = _o773.groupby("instrument", sort=False)["c_rawcombo"].transform(
        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(result.index)
    result["signal"] = result["c_rawcombo_s"]
    result["factor"] = result["signal"]''', 1)
S773 = S773.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "同伴残差与全维条件反转的原始合成源，做 span5 平滑（低换手取证版）；暴露表错一天。", 1)
(CODE / "773_合成源平滑span5纯源.py").write_text(S773, encoding="utf-8"); print("写出 773")
