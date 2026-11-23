# -*- coding: utf-8 -*-
"""10-20："先扩知识面再优化"——把信号库第一到四批里方向稳定的机制并进 1622（机制分组等风险合成，D-c）。
新组（组内截面秩等权，符号按经济先验 = 单独五窗口一致的方向）：
  投机热度  −K_MAX20、−K_PVC20、−K_SKEW20、−K_REV20、−K_PH250、−K_VTS、−K_VOLTS、−K_CGO（日频过度反应）
  微观慢组  −MS_AC1_20、+MS_JUMP_20、−MS_VR_20（个股的分钟级价格常态：弹跳、跳跃、趋势）
  日频补涨  +P_DAY（昨日落后同伴 → 今日补涨）
  特质反转  +P_CUM（扣掉同伴的今日累计反转）放进锚点反转组
组间配权同 1622：锚点反转 50%，其余各组平分 50%。与 1610、1622 比五窗口五项。"""
import numpy as np
import c1_upgrades as U
import c52_mech_group as M
from c1_upgrades import G, CLS, CP, build, MET
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = M.W; K10 = M.K10
NEW = {"投机热度": [(-1, n) for n in ("K_MAX20", "K_PVC20", "K_SKEW20", "K_REV20", "K_PH250", "K_VTS", "K_VOLTS", "K_CGO")],
       "微观慢组": [(-1, "MS_AC1_20"), (1, "MS_JUMP_20"), (-1, "MS_VR_20")],
       "日频补涨": [(1, "P_DAY")]}
NP = {n: M.npy(n) for v in NEW.values() for _, n in v}
NP["P_CUM"] = M.npy("P_CUM")
def compose(ix, add=(), cum=False, only161=False):
    gr = M.groups(ix)
    gr.pop("盘口深层")
    if only161:
        gr["锚点反转"] = gr["锚点反转"][:1]
    if cum:
        gr["锚点反转"] = gr["锚点反转"] + [NP["P_CUM"]]
    for k in add:
        gr[k] = [s * NP[n] for s, n in NEW[k]]
    gs = {k: M.zs(np.mean(v, axis=0)) for k, v in gr.items()}
    w = {k: (0.5 if k == "锚点反转" else 0.5 / (len(gs) - 1)) for k in gs}
    return np.where(z, 0.0, sum(w[k] * gs[k] for k in gs))
VAR = [("1622（D-c）", {}), ("+投机热度", {"add": ["投机热度"]}), ("+微观慢组", {"add": ["微观慢组"]}),
       ("+日频补涨", {"add": ["日频补涨"]}), ("+特质反转（锚点组）", {"cum": True}),
       ("+三个新组", {"add": list(NEW)}), ("+三个新组 + 特质反转", {"add": list(NEW), "cum": True}),
       ("锚点只用 161 + 三个新组 + 特质反转", {"add": list(NEW), "cum": True, "only161": True}),
       ("锚点只用 161", {"only161": True})]
if __name__ == "__main__":
    U.DEAL = {"Y_sd5", "X_n5d"}
    ref = {wn: G.ev(np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP), wn) for wn, ix in W.items()}
    print("%-34s " % "1610" + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
    base = None
    for nm, kw in VAR:
        res = {wn: G.ev(compose(ix, **kw), wn) for wn, ix in W.items()}
        base = base or res
        print("%-34s " % nm + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[res[wn][m] for m in MET]) for wn in W), flush=True)
        for tag, R0 in (("  vs 1610", ref), ("  vs 1622", base)):
            st = sum(M.ok({m: res[wn][m] - R0[wn][m] for m in MET}) for wn in W)
            print("%-34s " % tag + " | ".join("%s %+.4f/%+.3f/%+.1f/%+.3f/%+.4f" % (wn, *[res[wn][m] - R0[wn][m] for m in MET]) for wn in W) + "  %d/5" % st, flush=True)
