# -*- coding: utf-8 -*-
"""F2b：1607（树，h2_c2pf）再叠线性的 CLOSEPOS20 锚，五窗口五项，Δ 相对 1607。树的输入里已有 CP 列，这里看线性叠加能否再压换手。
可选第二个参数：另一个树的 tag（如 c2pf_aux05），同时报它相对 1607 的五项。"""
import sys
import numpy as np
from c1_upgrades import G, CP, MET, OUT
from g_go import WINS, IXW
c = G.c; z = G.z930
W = ("2023", "2024", "2021", "2022", "2020")
def load(tag):
    return {wn: np.where(z, 0.0, np.load(OUT / ("h2_%s_%s.npy" % (tag, wn)))) for wn in W}
H = load("c2pf")
sd = {wn: np.nanmedian([np.nanstd(H[wn][c.date_id == d]) for d in np.unique(c.date_id[H[wn] != 0])[::40]]) for wn in W}
print("1607 截面标准差", {k: round(v, 4) for k, v in sd.items()}, flush=True)
ref = {wn: G.ev(H[wn], wn) for wn in W}
print("1607 参考 " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
def report(name, F):
    line, star = name, 0
    for wn in W:
        r = G.ev(F[wn], wn)
        d = {m: r[m] - ref[wn][m] for m in MET}
        ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
        star += ok
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
    print(line + "  %d/5" % star, flush=True)
for e in (0.03, 0.05, 0.08):
    report("+CP%.2f" % e, {wn: np.where(z, 0.0, H[wn] + e * sd[wn] / 0.289 * CP) for wn in W})
if len(sys.argv) > 1:
    A = load(sys.argv[1])
    report("树 %s" % sys.argv[1], A)
    sda = {wn: np.nanmedian([np.nanstd(A[wn][c.date_id == d]) for d in np.unique(c.date_id[A[wn] != 0])[::40]]) for wn in W}
    for e in (0.03, 0.05):
        report("树 %s +CP%.2f" % (sys.argv[1], e), {wn: np.where(z, 0.0, A[wn] + e * sda[wn] / 0.289 * CP) for wn in W})
