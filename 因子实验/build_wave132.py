# -*- coding: utf-8 -*-
"""第一百三十二批（把已验证的两条独立杠杆合到一起）：
    758 = 头部全维条件反转 + 2.0 倍中性化（757 配方 + neutralize 2.0）
    759 = 头部全维条件反转（新源权重加大到 0.35）
"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
S757 = (CODE / "757_头部全维条件反转.py").read_text(encoding="utf-8")

OLD_N = '    result["factor"] = neutralize("signal", CHARS4, 1.5)'
assert OLD_N in S757
S758 = S757.replace(OLD_N, '    result["factor"] = neutralize("signal", CHARS4, 2.0)', 1)
S758 = S758.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "头部全维条件反转（757 配方）+ 2.0 倍中性化（中性化力度是本地看不见的大杠杆，需公榜验证）；暴露表错一天。", 1)
(CODE / "758_头部全维条件反转二点零中性化.py").write_text(S758, encoding="utf-8"); print("写出 758")

OLD_B = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.20])'
assert OLD_B in S757
NEW_B = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],\n                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.35])'
S759 = S757.replace(OLD_B, NEW_B, 1)
S759 = S759.replace("合法头部（688 配方）但条件反转分量换成全维版（9 风格 + 换手 + 振幅 + 偏度 + 笔数 = 13 维）；暴露表错一天。",
                    "头部全维条件反转但新源权重加大到 0.35（探索 IC 与换手的平衡点）；暴露表错一天。", 1)
(CODE / "759_头部全维条件反转源035.py").write_text(S759, encoding="utf-8"); print("写出 759")
