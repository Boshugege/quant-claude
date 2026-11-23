# -*- coding: utf-8 -*-
"""1619／1620（复活 Y_sd5／X_n5d 的平台安全写法）的逐行核对：评测机 2024 全年 factor_raw 与 engine 参考逐截面秩相关。
参考：1619 = 1603 平台口径 + 0.5×(完整 − 平台)；1620 = 1612 完整口径（三个成交笔数根都活）；1621 = 1610 平台口径 + 0.25×(完整 − 平台)。用法：c47_verify_revive.py <编号> <任务号>"""
import sys
import numpy as np
import pandas as pd
import c1_upgrades as U
from c1_upgrades import G, CLS, CP, build, MET
from g_go import inst_codes
num, job = sys.argv[1], sys.argv[2]
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
K12 = 0.544 / 0.289
def b(kind, full):
    U.DEAL = set() if full else {"Y_sd5", "X_n5d"}
    x = build(kind, 0, drop_deal=not full)
    if kind == "1601":
        x = x + 0.01 * K12 * CLS + 0.02 * K12 * CLS * (slot == 1430) + 0.03 * K12 * CP
    if kind == "1602":
        K10 = 0.2774 / 0.289
        x = x + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP
    return np.where(z, 0.0, x)
if num == "1619":
    P = b("1603", False); ref = P + 0.5 * (b("1603", True) - P)
elif num == "1622":
    import c52_mech_group as M                                               # 机制分组等风险合成 D-c 的 engine 参考
    ref = M.compose(0, "c")
elif num == "1621":
    P = b("1602", False); ref = P + 0.25 * (b("1602", True) - P)
else:
    ref = b("1601", True)
kd = pd.to_datetime(c.key_date); mk = np.asarray(kd.year == 2024)
R = pd.DataFrame({"date": kd[mk], "instrument": inst_codes(c)[np.flatnonzero(mk)], "ref": ref[mk]})
a = pd.read_parquet("/home/user/jyc_scorer/jobs/%s/factor_raw.parquet" % job)
a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
x = R.merge(a, on=["date", "instrument"]); x = x[x["date"].dt.strftime("%H%M") != "0930"]
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
print("%s %s 逐截面秩相关 均值 %.5f 中位 %.5f 最小 %.5f；< 0.98 的截面 %d／%d" % (num, job, rc.mean(), rc.median(), rc.min(), (rc < 0.98).sum(), len(rc)), flush=True)
r = G.ev(ref, "2024"); print("参考 2024 五项 %s" % " / ".join("%.4f" % r[k] for k in MET), flush=True)
