# -*- coding: utf-8 -*-
"""Claude 夜间 C4：1363（平台一致：只留六个安全根）的根权重剂量曲线 ×1／×1.5／×2／×2.5（± CLOSEPOS20 0.05）。
公榜已知：1363 0.82454、×1.25（1378）0.82727、×1.5（1379）0.82927。看 ×2 起是否还值得（明天第 5 个名额的候选）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c1_upgrades import G, CP, ax, W1363, DEAL, MET  # noqa: E402
from g_go import WINS, IXW  # noqa: E402
import numpy as np  # noqa: E402

z930 = G.z930
SAFE = [r for r in W1363 if r not in DEAL]


def build(ix, k, cp):
    base = np.asarray(G.B[("1363", ix)], float).copy()
    head = base - sum(W1363[r] * G.axis(r, ix) for r in W1363)
    B = head + sum(k * W1363[r] * ax(r, ix) for r in SAFE)
    if cp:
        B = B + cp * 0.50 / 0.289 * CP
    B[z930] = 0.0
    return B


ref = None
for k in (1.0, 1.5, 2.0, 2.5):
    for cp in (0.0, 0.05):
        Bs = {ix: build(ix, k, cp) for ix in (0, 1, 2)}
        r = {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn, _, _ in WINS}
        if ref is None:
            ref = r
        line = "×%.2f CP%.2f" % (k, cp)
        for wn in ("2023", "2024", "2021", "2022", "2020"):
            m = r[wn]
            d = {q: m[q] - ref[wn][q] for q in MET}
            line += " | %s %.4f/%.3f/%.4f Δ %+.5f/%+.4f/%+.2f/%+.4f/%+.4f" % (wn, m["ic_mean"], m["ic_ir"], m["turnover"],
                                                                         d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"])
        print(line, flush=True)
