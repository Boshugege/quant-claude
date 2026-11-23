# -*- coding: utf-8 -*-
"""R5：把有真实平台分数的 85（放量加权反转 0.742）、86（小单主导加权反转 0.721）接进 104 结构；
以及更长滚动窗口的阶梯、中性化力度 × 振幅的组合。"""
import numpy as np
import pandas as pd

import factor_lib as fl
from factor_lib import grolling, neutral4, rank_blend, rank_weighted
from cand_r1 import downside_share
from cand_r2 import peer_res_ladder_w
from cand_r3 import base_104, base_soft2, blend, n_amp


def w85(p):
    """85 号：放量加权反转（相对放量越大，反转信号权重越低）。"""
    return rank_weighted(p, "r_rv", "w_relvol", invert=False)


def w86(p):
    """86 号：小单主导加权反转（相对单笔规模越大，反转信号权重越低）。"""
    return rank_weighted(p, "r_rv", "w_relsize", invert=True)


def soft2_frac(f):
    return lambda p: neutral4(p, peer_res_ladder_w(p, mode="soft", power=2.0, k=100), f)


def ladder_rv16(p, parts=("30", "rv4", "rv8", "rv16", "day")):
    return fl.ladder(p, list(parts), False)


def _multi(*pairs):
    def f(p):
        cols, ws = {}, []
        for i, (fn, w) in enumerate(pairs):
            cols["c%d" % i] = fn(p)
            ws.append(w)
        return rank_blend(p.assign(**cols), list(cols), ws)
    return f


CANDS = {
    # 85 / 86 接入 104 结构
    "104_w85_0.2": blend(base_104, w85, 0.2),
    "104_w85_0.3": blend(base_104, w85, 0.3),
    "104_w86_0.2": blend(base_104, w86, 0.2),
    "104_w86_0.3": blend(base_104, w86, 0.3),
    "soft2_w85_0.2": blend(base_soft2, w85, 0.2),
    "soft2_w86_0.2": blend(base_soft2, w86, 0.2),
    "soft2_ds.15_amp.15_w86.2": _multi((base_soft2, 1.0), (downside_share, 0.15), (n_amp, 0.15), (w86, 0.2)),
    # 更长滚动窗口
    "ref_90_rv16": ladder_rv16,
    "peer_rv16": lambda p: neutral4(p, fl.peer_res_ladder(p, parts=("30", "rv4", "rv8", "rv16", "day")), 1.5),
    # 中性化力度 × 振幅
    "soft2_f2.0_amp.20": _multi((soft2_frac(2.0), 1.0), (n_amp, 0.20)),
    "soft2_f2.5_amp.20": _multi((soft2_frac(2.5), 1.0), (n_amp, 0.20)),
    "104_amp.20_ds.15": _multi((base_104, 1.0), (n_amp, 0.20), (downside_share, 0.15)),
}
