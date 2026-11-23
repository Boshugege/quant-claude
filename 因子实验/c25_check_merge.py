# -*- coding: utf-8 -*-
"""核对合并模型：用合并 JSON 在 engine 面板上打分，与两个模型 npy 的平均逐行比较（2023 窗口）。"""
import json, sys
import numpy as np
import h2_robust as H
from g_go import GO
from c1_upgrades import OUT
J = json.load(open(sys.argv[1], encoding="utf-8")); tags = sys.argv[2:]
F, E, P = J["feat_order"], [np.asarray(e, np.float64) for e in J["edges"]], J["model"]["params"]
G = GO(); c = G.c; z = c.slot_of_date[c.date_id] == 930
H.EXTRA[:] = ["L_5", "VSH20", "PACE20", "DOW10", "CLS20", "CLOSEPOS20"]; H.DROP[:] = ["N_dir", "Y_sd5", "X_n5d"]
H.PF1363[0] = True; H.H1363_SUB[:] = ["Y_sd5", "X_n5d"]
X = H.build_X(G, 0, extra=H.EXTRA, slots=True)
assert X.shape[1] == len(F), (X.shape, len(F))
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
s = np.where(z, 0.0, s)
avg = np.mean([np.load(OUT / ("h2_%s_2023.npy" % t)) for t in tags], axis=0)
mk = avg != 0
print("合并模型 vs 两模型平均（2023 行）：最大差 %.2e，相关 %.6f" % (np.abs(s[mk] - avg[mk]).max(), np.corrcoef(s[mk], avg[mk])[0, 1]), flush=True)
r = G.ev(s, "2023"); print("合并模型 2023 五项", " / ".join("%.4f" % r[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")))
