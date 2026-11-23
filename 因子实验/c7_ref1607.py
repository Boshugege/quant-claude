# -*- coding: utf-8 -*-
"""1607 代码版核对：用 h2_c2_pf.json 在"代码等价"的输入上预测（H1363 用含成交笔数根的完整原始值，与代码版在本地算出来的一致），
与评测机 c1607（2024 不分块）的 factor_raw 比逐截面秩相关与 2024 五项。用法：python c7_ref1607.py <任务号>"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import h2_robust as H  # noqa: E402
import h2_stack as HS  # noqa: E402
from g_go import GO, inst_codes  # noqa: E402

job = sys.argv[1]
J = json.load(open(pathlib.Path(__file__).resolve().parent / "models" / "h2_c2_pf.json"))
H.DROP.extend(["N_dir", "Y_sd5", "X_n5d"])
H.PF1363[0] = False                                                     # 代码等价：本地 H1363 含成交笔数根
H.EXTRA = list(J["extra"]) if hasattr(H, "EXTRA") else None
G = GO()
c = G.c
X = H.build_X(G, 0, extra=J["extra"])
assert X.shape[1] == len(J["feat_order"]), (X.shape, len(J["feat_order"]))
E = np.asarray(J["edges"], np.float32)
B = HS.to_bins(X, E)
out = np.full(len(X), J["model"]["base"])
for tr in J["model"]["trees"]:
    LV, SF, SK, SD, HSp = HS._tree_tables(tr)
    node = np.zeros(len(B), np.int64)
    depth = int(SD[HSp].max()) if HSp.any() else -1
    for d in range(depth + 1):
        cur = HSp[node] & (SD[node] == d)
        if not cur.any():
            break
        ii = np.flatnonzero(cur)
        node[ii] = np.where(B[ii, SF[node[ii]]] <= SK[node[ii]], 2 * node[ii] + 1, 2 * node[ii] + 2)
    out += J["model"]["params"]["lr"] * LV[node]
out[G.z930] = 0.0
kd = pd.to_datetime(c.key_date)
m = kd.year == 2024
ref = pd.DataFrame({"date": kd[m], "instrument": inst_codes(c)[np.flatnonzero(m)], "ref": out[m]})
a = pd.read_parquet("/home/user/jyc_scorer/jobs/%s/factor_raw.parquet" % job)
a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
x = ref.merge(a, on=["date", "instrument"]); x = x[x["date"].dt.strftime("%H%M") != "0930"]
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
print("1607 vs 参考 逐截面秩相关 均值 %.5f 最小 %.5f 行 %d" % (rc.mean(), rc.min(), len(x)), flush=True)
mr = G.ev(out, "2024")
print("参考 2024 五项 %s" % " / ".join("%.4f" % mr[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")), flush=True)
