# -*- coding: utf-8 -*-
"""Claude 夜间 C1b：队友头部（s152／s155／s161）+ 我们的 6 个平台安全根（去掉成交笔数类）+ 三小根 + CLOSEPOS20 锚，口径 engine.Core.evaluate。
看 s152 底座（公榜已知 0.84357、不用成交笔数）上这一套的五窗口五项，作为明天"队友头部 + 我们的根"方向的候选。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
from c1_upgrades import G, CP, ax, W1363, SMALL, DEAL, MET, est  # noqa: E402
from g_go import WINS, IXW  # noqa: E402

z930 = G.z930
SAFE = [r for r in W1363 if r not in DEAL]
NAMES = {"s152": ("fm_s152", "o21_s152", "o20_s152"), "s155": ("fm_s155", "o21_s155", "o20_s155"), "s161": ("fm_s161", "o21_s161", "o20_s161")}
RAW = {}


def raw(h, ix):
    if (h, ix) not in RAW:
        x = np.asarray(R.load(NAMES[h][ix])[0], float).copy()
        x[z930] = 0.0
        RAW[(h, ix)] = x
    return RAW[(h, ix)]


def build(h, ix, roots=True, cp=0.0):
    B = raw(h, ix).copy()
    if roots:
        for r in SAFE:
            B += W1363[r] * ax(r, ix)
        for k, w in SMALL.items():
            B += w * G.axis(k, ix)
    if cp:
        sd = np.nanmedian([np.nanstd(B[(G.c.date_id == d)]) for d in np.unique(G.c.date_id)[::50]])
        B += cp * sd / 0.289 * CP
    B[z930] = 0.0
    return B


for h in ("s152", "s155", "s161"):
    res = {}
    for name, kw in (("底座", dict(roots=False)), ("+6根+三小根", dict()), ("+6根+三小根+CP0.05", dict(cp=0.05))):
        Bs = {ix: build(h, ix, **kw) for ix in (0, 1, 2)}
        res[name] = {wn: {k: G.ev(Bs[IXW[wn]], wn)[k] for k in MET} for wn, _, _ in WINS}
    for name in res:
        line = "%-5s %-18s" % (h, name)
        for wn in ("2023", "2024", "2021", "2022", "2020"):
            m = res[name][wn]
            if name == "底座":
                line += " | %s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"])
            else:
                b = res["底座"][wn]
                d = {k: m[k] - b[k] for k in MET}
                line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"])
        print(line, flush=True)
