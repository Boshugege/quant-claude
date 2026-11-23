# -*- coding: utf-8 -*-
"""树方向的正确对照：已交的 1600（h2_seas_gbdt_100.json）在平台上的真实表现 = Y_sd5、X_n5d 两列为 0（跨日 w5 写法在平台上失效）、
H1363 里扣掉这两根（N_dir 活着，保留）。与 1607（c2pf，三个都去）、1607N（c2n，只去 Y+X）比五窗口五项。
先用未改动的特征复现 h2_seas npy，确认打分实现正确。"""
import json
import sys
import numpy as np
import h2_robust as H
from g_go import GO
from c1_upgrades import OUT, W1363
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
W = (("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2))
J = json.load(open("/home/user/quant-claude/因子实验/models/h2_seas_gbdt_100.json", encoding="utf-8"))
F, E = J["feat_order"], [np.asarray(e, np.float64) for e in J["edges"]]
P = J["model"]["params"]
G = GO(); c = G.c; z = c.slot_of_date[c.date_id] == 930
H.EXTRA[:] = list(J.get("extra") or ["L_5", "VSH20", "PACE20", "DOW10"]); H.DROP[:] = []; H.PF1363[0] = False
def score(X):
    B = np.empty(X.shape, np.int64)
    for j in range(X.shape[1]):
        B[:, j] = np.searchsorted(E[j], X[:, j], side="left")
    s = np.full(len(X), float(J["model"]["base"])); ar = np.arange(len(X))
    for tr in J["model"]["trees"]:
        m = max(int(it[1]) for it in tr) + 1
        LV, SF, SK, HS = np.zeros(m), np.zeros(m, np.int64), np.zeros(m, np.int64), np.zeros(m, bool)
        for it in tr:
            if it[0] == "split":
                SF[int(it[1])], SK[int(it[1])], HS[int(it[1])] = int(it[2]), int(it[3]), True
            else:
                LV[int(it[1])] = float(it[2])
        node = np.zeros(len(X), np.int64)
        for _ in range(int(P["depth"])):
            hs = HS[node]
            if not hs.any():
                break
            left = B[ar, SF[node]] <= SK[node]
            node = np.where(hs, np.where(left, 2 * node + 1, 2 * node + 2), node)
        s += float(P["lr"]) * LV[node]
    return s
TAGS = [(t, t) for t in sys.argv[1:]] or [("c2pf", "1607"), ("c2n", "1607N"), ("c2pf_aux05", "aux05")]
print("特征", len(F), F, flush=True)
res = {}
for ix in (0, 1, 2):
    X = H.build_X(G, ix, extra=H.EXTRA, slots=True)
    assert X.shape[1] == len(F), (X.shape, len(F))
    full = np.where(z, 0.0, score(X))
    Xp = X.copy()
    for lab in ("Y_sd5", "X_n5d"):
        Xp[:, F.index(lab)] = 0.0
    Xp[:, F.index("H1363")] -= sum(W1363[l] * G.axis(l, ix) for l in ("Y_sd5", "X_n5d"))
    plat = np.where(z, 0.0, score(Xp))
    for wn, wi in W:
        if wi != ix:
            continue
        ref = np.load(OUT / ("h2_seas_%s.npy" % wn))
        mk = ref != 0
        print("%s 复现 h2_seas：秩相关 %.5f" % (wn, np.corrcoef(np.argsort(np.argsort(ref[mk])), np.argsort(np.argsort(full[mk])))[0, 1]), flush=True)
        res[wn] = {"1600本地": G.ev(full, wn), "1600平台": G.ev(plat, wn)}
        np.save(OUT / ("h2_seasplat_%s.npy" % wn), plat)
        for tag, nm in TAGS:
            p = OUT / ("h2_%s_%s.npy" % (tag, wn))
            if p.exists():
                res[wn][nm] = G.ev(np.where(z, 0.0, np.load(p)), wn)
    del X, Xp
for nm in ["1600本地", "1600平台"] + [n for _, n in TAGS]:
    if all(nm in res[wn] for wn, _ in W):
        print("%-8s " % nm + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[res[wn][nm][m] for m in MET]) for wn, _ in W), flush=True)
for nm in [n for _, n in TAGS]:
    if all(nm in res[wn] for wn, _ in W):
        line, star = "%s vs 1600平台" % nm, 0
        for wn, _ in W:
            d = {m: res[wn][nm][m] - res[wn]["1600平台"][m] for m in MET}
            ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            star += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % star, flush=True)
