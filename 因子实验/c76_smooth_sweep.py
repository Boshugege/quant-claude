# -*- coding: utf-8 -*-
"""纯方向深挖 J6（10-20）：平滑的系统扫描。c70 发现跨截面指数平滑让快的反转信号五项同时变好；这里把之前所有快信号都扫一遍，
看哪些机制"累积冲击"比"单段冲击"更有信息。每个信号取单独五窗口一致的方向（符号在名字后），α ∈ {0, 0.3, 0.5, 0.7}。"""
import os
import numpy as np
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, MET, OUT
c = G.c; z = G.z930; did = c.date_id
W = ("2023", "2024", "2021", "2022", "2020")
SIG = [("MS_AC1", -1), ("MS_VR", -1), ("MS_SMART", 1), ("MS_JUMP", 1), ("MS_AMI", 1), ("I_INTRA", -1), ("CF_DIV", 1),
       ("QL_ARR", 1), ("QL_LATE", 1), ("QL_SLATE", -1), ("QL_NIMB23", 1), ("OQ_SIZE", -1), ("OQ_DSIZE", -1), ("P_LAG", 1), ("CF_OFI3", -1)]
SIG += [(s, int(v)) for s, v in (x.split(":") for x in os.environ.get("EXTRA_SIG", "").split(",") if x)]
for nm, sg in SIG:
    p = OUT / ("h1_%s.npy" % nm)
    if not p.exists():
        print("缺", nm, flush=True); continue
    x = np.load(p).astype(float)
    r0 = np.nan_to_num(O.rc_of(np.where(x != 0, x, np.nan), did, c.n_dates), nan=0.0) * sg
    r0[z] = 0.0
    for a in (0.0, 0.3, 0.5, 0.7):
        f = H.ewma_col(r0, c.inst, a) if a > 0 else r0
        f = np.where(z, 0.0, f)
        r = {wn: G.ev(f, wn) for wn in W}
        print("%-10s %+d α=%.1f " % (nm, sg, a) + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
