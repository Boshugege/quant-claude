# -*- coding: utf-8 -*-
"""树代码版的逐行核对：用模型 JSON 在 engine 面板上打分作参考。注意本地代码版里 1363 的 Y_sd5、X_n5d 是活的（本地有数据），
所以参考的 H1363 用完整的 1363（不扣），其余特征与训练相同（去 N_dir／Y_sd5／X_n5d 三列）。用法：c5h_verify_tree.py <模型.json> <任务号> <编号>"""
import json, sys
import numpy as np
import pandas as pd
import h2_robust as H
from g_go import GO, inst_codes
fj, job, num = sys.argv[1], sys.argv[2], sys.argv[3]
J = json.load(open(fj, encoding="utf-8"))
F, E, P = J["feat_order"], [np.asarray(e, np.float64) for e in J["edges"]], J["model"]["params"]
G = GO(); c = G.c; z = c.slot_of_date[c.date_id] == 930
H.EXTRA[:] = ["L_5", "VSH20", "PACE20", "DOW10", "CLS20", "CLOSEPOS20"]; H.DROP[:] = ["N_dir", "Y_sd5", "X_n5d"]; H.PF1363[0] = False
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
        node = np.where(hs, np.where(B[ar, SF[node]] <= SK[node], 2 * node + 1, 2 * node + 2), node)
    s += float(P["lr"]) * LV[node]
s = np.where(z, 0.0, s)
kd = pd.to_datetime(c.key_date); mk = (kd.year == 2024)
ref = pd.DataFrame({"date": kd[mk], "instrument": inst_codes(c)[np.flatnonzero(mk)], "ref": s[mk]})
a = pd.read_parquet("/home/user/jyc_scorer/jobs/%s/factor_raw.parquet" % job)
a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
x = ref.merge(a, on=["date", "instrument"]); x = x[x["date"].dt.strftime("%H%M") != "0930"]
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
print("%s %s 逐截面秩相关 均值 %.5f 中位 %.5f 最小 %.5f；< 0.98 的截面 %d／%d" % (num, job, rc.mean(), rc.median(), rc.min(), (rc < 0.98).sum(), len(rc)), flush=True)
r = G.ev(s, "2024"); print("参考 2024 五项 %s" % " / ".join("%.4f" % r[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")), flush=True)
