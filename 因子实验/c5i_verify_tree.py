# -*- coding: utf-8 -*-
"""c5i（10-20，1627 起）：同 c5h，但额外特征从环境变量 EXTRA 读（逗号分隔），评分机输出用文件路径给出（可在 macstudio 上跑）。
用法：EXTRA=L_5,…,OQ_DSIZE_20 c5i_verify_tree.py <模型.json> <factor_raw.parquet 路径> <编号> [年份]
原 c5h 说明：树代码版的逐行核对：用模型 JSON 在 engine 面板上打分作参考。注意本地代码版里 1363 的 Y_sd5、X_n5d 是活的（本地有数据），
所以参考的 H1363 用完整的 1363（不扣），其余特征与训练相同（去 N_dir／Y_sd5／X_n5d 三列）。用法：c5h_verify_tree.py <模型.json> <任务号> <编号>"""
import json, os, sys
import numpy as np
import pandas as pd
import h2_robust as H
from g_go import GO, inst_codes
fj, job, num = sys.argv[1], sys.argv[2], sys.argv[3]
YEAR = int(sys.argv[4]) if len(sys.argv) > 4 else 2024
J = json.load(open(fj, encoding="utf-8"))
F, E, P = J["feat_order"], [np.asarray(e, np.float64) for e in J["edges"]], J["model"]["params"]
G = GO(); c = G.c; z = c.slot_of_date[c.date_id] == 930
H.EXTRA[:] = os.environ["EXTRA"].split(","); H.DROP[:] = ["N_dir", "Y_sd5", "X_n5d"]; H.PF1363[0] = False
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
kd = pd.to_datetime(c.key_date); mk = (kd.year == YEAR)
ref = pd.DataFrame({"date": kd[mk], "instrument": inst_codes(c)[np.flatnonzero(mk)], "ref": s[mk]})
a = pd.read_parquet(job)
a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
x = ref.merge(a, on=["date", "instrument"]); x = x[x["date"].dt.strftime("%H%M") != "0930"]
print("对齐行数 %d（参考 %d、评分机 %d）" % (len(x), len(ref), len(a)), flush=True)
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
print("%s %s 逐截面秩相关 均值 %.5f 中位 %.5f 最小 %.5f；< 0.98 的截面 %d／%d" % (num, job, rc.mean(), rc.median(), rc.min(), (rc < 0.98).sum(), len(rc)), flush=True)
r = G.ev(s, str(YEAR))
print("参考 %d 五项 %s" % (YEAR, " / ".join("%.4f" % r[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"))), flush=True)
