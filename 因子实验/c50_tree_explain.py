# -*- coding: utf-8 -*-
"""1617 的可解释性分析（10-19）：
(1) 每个特征被用来分叉的次数占比（16 个模型、1920 棵树合计）；
(2) 单特征敏感度：2024 年随机 20 万行，把某个特征在截面排名上 +0.1／−0.1（原始值特征按其截面标准差的 0.25 倍），
    看模型输出的平均变化（方向是否符合经济直觉），以及与该特征的边际关系是否单调；
(3) 时段交互：同一特征 +0.1 的敏感度按 7 个时段分开看。"""
import json, sys
import numpy as np
import pandas as pd
import h2_robust as H
from g_go import GO
J = json.load(open(sys.argv[1], encoding="utf-8"))
F, E, P = J["feat_order"], [np.asarray(e, np.float64) for e in J["edges"]], J["model"]["params"]
trees, base, lr = J["model"]["trees"], float(J["model"]["base"]), float(P["lr"])
cnt = np.zeros(len(F))
for tr in trees:
    for it in tr:
        if it[0] == "split":
            cnt[int(it[2])] += 1
print("分叉次数占比（前 15）")
for j in np.argsort(-cnt)[:15]:
    print("   %-12s %.3f" % (F[j], cnt[j] / cnt.sum()))
G = GO(); c = G.c
H.EXTRA[:] = ["L_5", "VSH20", "PACE20", "DOW10", "CLS20", "CLOSEPOS20"]; H.DROP[:] = ["N_dir", "Y_sd5", "X_n5d"]; H.PF1363[0] = True; H.H1363_SUB[:] = ["Y_sd5", "X_n5d"]
X = H.build_X(G, 0, extra=H.EXTRA, slots=True)
kd = pd.to_datetime(c.key_date); slot = c.slot_of_date[c.date_id]
rows = np.flatnonzero(np.asarray(kd.year == 2024) & (slot != 930))
rng = np.random.default_rng(0); rows = np.sort(rng.choice(rows, 200000, replace=False))
X0 = X[rows].astype(np.float64); sl = slot[rows]
def score(Xs):
    B = np.empty(Xs.shape, np.int64)
    for j in range(Xs.shape[1]):
        B[:, j] = np.searchsorted(E[j], Xs[:, j], side="left")
    s = np.full(len(Xs), base); ar = np.arange(len(Xs))
    for tr in trees:
        m = max(int(it[1]) for it in tr) + 1
        LV, SF, SK, HS = np.zeros(m), np.zeros(m, np.int64), np.zeros(m, np.int64), np.zeros(m, bool)
        for it in tr:
            if it[0] == "split":
                SF[int(it[1])], SK[int(it[1])], HS[int(it[1])] = int(it[2]), int(it[3]), True
            else:
                LV[int(it[1])] = float(it[2])
        node = np.zeros(len(Xs), np.int64)
        for _ in range(int(P["depth"])):
            hs = HS[node]
            if not hs.any():
                break
            node = np.where(hs, np.where(B[ar, SF[node]] <= SK[node], 2 * node + 1, 2 * node + 2), node)
        s += lr * LV[node]
    return s
s0 = score(X0); sd0 = s0.std()
print("\n单特征敏感度（2024 年 20 万行；变化以模型输出标准差为单位；正 = 该特征越大，因子越大）")
SLOTS = [1000, 1030, 1100, 1130, 1330, 1400, 1430]
for j in np.argsort(-cnt)[:15]:
    nm = F[j]
    if nm.startswith("slot"):
        continue
    step = 0.1 if np.nanstd(X0[:, j]) < 0.35 else 0.25 * np.nanstd(X0[:, j])
    up = X0.copy(); up[:, j] += step; dn = X0.copy(); dn[:, j] -= step
    du, dd = (score(up) - s0) / sd0, (score(dn) - s0) / sd0
    per = " ".join("%d:%+.3f" % (t, du[sl == t].mean()) for t in SLOTS)
    print("   %-12s 上推 %+.3f 下推 %+.3f（单调 %s）| 分时段上推 %s" % (nm, du.mean(), dd.mean(), "是" if du.mean() * dd.mean() < 0 else "否", per), flush=True)
