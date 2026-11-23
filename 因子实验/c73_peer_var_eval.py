# -*- coding: utf-8 -*-
"""纯方向深挖 J4 评估（10-20）：同伴定义变体（K 个同伴、LB 日相关）下的特质段反转（−秩(D_R − P_SEG)）与特质日内反转（秩(P_CUM)），五窗口五项。"""
import numpy as np
import d2_optimize as O
from c1_upgrades import G, MET, OUT
c = G.c; z = G.z930; did = c.date_id
W = ("2023", "2024", "2021", "2022", "2020")
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask):
    r = np.nan_to_num(O.rc_of(np.where(mask, x, np.nan), did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
DR = raw("D_R"); mD = DR != 0
for suf in ("", "_k10l60", "_k50l60", "_k20l20", "_k20l120"):
    ps, pc = raw("P_SEG" + suf), raw("P_CUM" + suf)
    for nm, x in (("特质段反转", -rk(DR - ps, mD & (ps != 0))), ("特质日内反转", rk(pc, pc != 0))):
        r = {wn: G.ev(x, wn) for wn in W}
        print("%-10s %-10s " % (nm, suf or "_k20l60") + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
