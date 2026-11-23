# -*- coding: utf-8 -*-
"""F6：剂量方向与全局定权方向的"五项都不差"升级。
 (1) 1611 候选 = 1379（公榜 0.82927：1363 的九根 ×1.5）+ CLS20／CP 锚对，Δ 相对 1379；
 (2) 1601／1603（公榜已知，已含 CLS 0.05/0.10）再加 CP 与少量 CLS，Δ 相对各自。
 两个底座口径：去 Y+X（N_dir 活）与三个都去。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CP, CLS, build, W1363, MET
from g_go import WINS, IXW
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = ("2023", "2024", "2021", "2022", "2020")
def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn in W}
def rep(name, ref, Bs):
    r = ev5(Bs); line, star = name, 0
    for wn in W:
        d = {m: r[wn][m] - ref[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
def b1379(ix, dset):
    B = np.asarray(G.B[("1363", ix)], float).copy()
    for r in W1363:                                                  # 1379 = 1363 的九根权重 ×1.5（死根照样 ×1.5，平台上为零）
        B += (0.5 * W1363[r] - (1.5 * W1363[r] if r in dset else 0.0)) * G.axis(r, ix)
    B[z] = 0.0
    return B
for dname, dset in (("去Y+X", {"Y_sd5", "X_n5d"}), ("三个都去", {"N_dir", "Y_sd5", "X_n5d"})):
    U.DEAL = dset
    sets = {"1379": ({ix: b1379(ix, dset) for ix in (0, 1, 2)}, 0.50 / 0.289,
                     ((0.05, 0.10, 0.05), (0.05, 0.10, 0.075), (0.03, 0.06, 0.05))),
            "1601": ({ix: build("1601", ix, drop_deal=True) for ix in (0, 1, 2)}, 0.544 / 0.289,
                     ((0.0, 0.05, 0.05), (0.02, 0.04, 0.05), (0.0, 0.05, 0.075), (0.025, 0.05, 0.075))),
            "1603": ({ix: build("1603", ix, drop_deal=True) for ix in (0, 1, 2)}, 0.655 / 0.289,
                     ((0.0, 0.05, 0.05), (0.02, 0.04, 0.05), (0.0, 0.05, 0.075), (0.025, 0.05, 0.075)))}
    for kind, (base, K, vs) in sets.items():
        ref = ev5(base)
        print("== %s %s 底座 " % (kind, dname) + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
        for a, b, e in vs:
            Bs = {}
            for ix in (0, 1, 2):
                x = base[ix] + a * K * CLS + b * K * CLS * (slot == 1430) + e * K * CP
                x[z] = 0.0
                Bs[ix] = x
            rep("%s %s +CLS%.3f+%.3f CP%.3f" % (kind, dname, a, b, e), ref, Bs)
