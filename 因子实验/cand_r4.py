# -*- coding: utf-8 -*-
"""R4：把 R3 的两个赢家（下行波动占比、振幅）与软同伴组合，做权重扫描；并尝试 09:30 截面专用增强。"""
import numpy as np
import pandas as pd

import factor_lib as fl
from factor_lib import grolling, neutral4, rank_blend
from cand_r1 import d5_rev, downside_share, vol_centroid
from cand_r2 import peer_res_ladder_w
from cand_r3 import base_104, base_soft2, blend, n_amp, n_gap, n_iskew


def _multi(*pairs):
    """按 (基数, 附加函数, 权重) 列表做截面排名加权合成。"""
    def f(p):
        cols, ws = {}, []
        for i, (fn, w) in enumerate(pairs):
            cols["c%d" % i] = fn(p)
            ws.append(w)
        return rank_blend(p.assign(**cols), list(cols), ws)
    return f


def s930_enh(p, base_fn=base_104):
    """09:30 截面改用专用组合（两小时/四小时 VWAP 偏离 + 隔夜反转 + 5 日反转 + 下行波动占比）。"""
    alt = neutral4(p, rank_blend(p.assign(a=fl.rv_raw(p, 4), b=p["r_day"], c=d5_rev(p),
                                          d=downside_share(p), e=fl.rv_raw(p, 8)),
                                ["a", "b", "c", "d", "e"]), 1.5)
    return base_fn(p).where(p["slot"] != 930, alt)


def s930_enh2(p):
    """09:30 截面只用两小时 VWAP 偏离 + 下行波动占比（更少分量，避免噪声）。"""
    alt = neutral4(p, rank_blend(p.assign(a=fl.rv_raw(p, 4), b=downside_share(p)), ["a", "b"]), 1.5)
    return base_104(p).where(p["slot"] != 930, alt)


CANDS = {
    # 振幅权重扫描（R3 里 IC_IR/压力最高）
    "104_amp0.05": blend(base_104, n_amp, 0.05),
    "104_amp0.10": blend(base_104, n_amp, 0.10),
    "104_amp0.15": blend(base_104, n_amp, 0.15),
    "104_amp0.20": blend(base_104, n_amp, 0.20),
    "104_amp0.30": blend(base_104, n_amp, 0.30),
    # 下行波动 + 振幅
    "104_ds.15_amp.15": _multi((base_104, 1.0), (downside_share, 0.15), (n_amp, 0.15)),
    "104_ds.15_amp.10": _multi((base_104, 1.0), (downside_share, 0.15), (n_amp, 0.10)),
    "104_ds.10_amp.15": _multi((base_104, 1.0), (downside_share, 0.10), (n_amp, 0.15)),
    "soft2_amp.15": blend(base_soft2, n_amp, 0.15),
    "soft2_ds.15_amp.15": _multi((base_soft2, 1.0), (downside_share, 0.15), (n_amp, 0.15)),
    "soft2_ds.15_amp.10": _multi((base_soft2, 1.0), (downside_share, 0.15), (n_amp, 0.10)),
    # 换手率更低的变体
    "104_ds.15_gap.15": _multi((base_104, 1.0), (downside_share, 0.15), (n_gap, 0.15)),
    "104_ds.15_cent.15": _multi((base_104, 1.0), (downside_share, 0.15), (vol_centroid, 0.15)),
    # 09:30 专用增强
    "s930_enh": s930_enh,
    "s930_enh_k4": lambda p: s930_enh(p, base_soft2),
    "s930_enh2": s930_enh2,
}
