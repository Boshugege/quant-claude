# -*- coding: utf-8 -*-
"""第七批公榜换手更接近"含成交笔数根"的本地版本 → 检查哪个成交笔数类根在平台上还活着。
四种口径：全含（已交）、只去 Y_sd5+X_n5d（N_dir 活）、只去 N_dir、三个都去（平台一致）。2023/2024 两年平均的五项。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, build, W1363, MET
z = G.z930
def ev(B):
    B = np.asarray(B, float).copy(); B[z] = 0.0
    r = [G.ev(B, wn) for wn in ("2023", "2024")]
    return {m: (r[0][m] + r[1][m]) / 2 for m in MET}
SETS = {"全含": set(), "去Y_sd5+X_n5d": {"Y_sd5", "X_n5d"}, "去N_dir": {"N_dir"}, "三个都去": {"N_dir", "Y_sd5", "X_n5d"}}
for kind in ("1363", "1602", "1601", "1603"):
    for nm, dset in SETS.items():
        if kind == "1363":
            B = np.asarray(G.B[("1363", 0)], float) - sum(W1363[r] * G.axis(r, 0) for r in dset)
        else:
            U.DEAL = dset
            B = build(kind, 0, drop_deal=True)
        r = ev(B)
        print("%-5s %-14s IC %.4f IR %.4f 多空 %.2f 压力 %.4f 换手 %.4f" % (kind, nm, r["ic_mean"], r["ic_ir"], r["ls_sharpe"], r["stress"], r["turnover"]), flush=True)
