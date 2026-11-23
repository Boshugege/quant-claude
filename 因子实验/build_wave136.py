# -*- coding: utf-8 -*-
"""第一百三十六批（收口批次）：
    774 = 767（头部全维条件反转不平滑）+ 2.0 倍中性化
    775 = 全维条件反转 span5 平滑纯源（低换手取证版）
    776 = 头部（757 配方）但全维分量用 span5 平滑
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S767 = (CODE / "767_头部全维条件反转不平滑.py").read_text(encoding="utf-8")
S757 = (CODE / "757_头部全维条件反转.py").read_text(encoding="utf-8")

OLD_N = '    result["factor"] = neutralize("signal", CHARS4, 1.5)'
assert OLD_N in S767
S774 = S767.replace(OLD_N, '    result["factor"] = neutralize("signal", CHARS4, 2.0)', 1)
S774 = S774.replace("头部全维条件反转且该分量不做平滑（原始全维条件反转，探索 IC 上限）；暴露表错一天。",
                    "头部全维条件反转（不平滑）+ 2.0 倍中性化；暴露表错一天。", 1)
(CODE / "774_头部全维不平滑二点零中性化.py").write_text(S774, encoding="utf-8"); print("写出 774")

OLD_SIG = '''    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],
                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert OLD_SIG in S757
S775 = S757.replace(OLD_SIG, '''    _o775 = result.sort_values(["instrument", "date"])
    result["c_conds5"] = _o775.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(result.index)
    result["signal"] = result["c_conds5"]
    result["factor"] = result["signal"]''', 1)
S775 = S775.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "全维条件反转（13 维）span5 平滑纯源（低换手取证版）；暴露表错一天。", 1)
(CODE / "775_全维条件反转span5纯源.py").write_text(S775, encoding="utf-8"); print("写出 775")

OLD_C670 = '''    result["c_conds670"] = _ord670.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)'''
assert OLD_C670 in S757
S776 = S757.replace(OLD_C670, '''    result["c_conds670"] = _ord670.groupby("instrument", sort=False)["c_condrev"].transform(
        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(result.index)''', 1)
S776 = S776.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "头部（757 配方）但全维条件反转分量改用 span5 平滑；暴露表错一天。", 1)
(CODE / "776_头部全维分量span5.py").write_text(S776, encoding="utf-8"); print("写出 776")
