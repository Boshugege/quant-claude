# -*- coding: utf-8 -*-
"""值不值得复活 Y_sd5／X_n5d：它们在平台上为零只因跨日 5 日窗口的 SQL 写法失效（deal_number 本身完整）。
比较 1363 系三个方向的"完整口径"（三个成交笔数根都活）与"平台口径"（Y、X 为零），五窗口五项。若完整口径五项全好，
用平台安全写法（SQL 只算日汇总，5 日滚动放到 pandas）复活这两个根，就是三个方向的合规升级。"""
import numpy as np
import c1_upgrades as U
from c1_upgrades import G, CLS, CP, build, W1363, MET
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K12, K11 = 0.544 / 0.289, 0.50 / 0.289
def b1379(ix, dead):
    B = np.asarray(G.B[("1363", ix)], float).copy()
    for r in W1363:
        B += (0.5 * W1363[r] - (1.5 * W1363[r] if r in dead else 0.0)) * G.axis(r, ix)
    return B
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
DIRS = {
    "1603": lambda ix, full: (U.__setattr__("DEAL", set() if full else {"Y_sd5", "X_n5d"}), build("1603", ix, drop_deal=not full))[1],
    "1612": lambda ix, full: (U.__setattr__("DEAL", set() if full else {"Y_sd5", "X_n5d"}), build("1601", ix, drop_deal=not full) + 0.01 * K12 * CLS + 0.02 * K12 * CLS * (slot == 1430) + 0.03 * K12 * CP)[1],
    "1611": lambda ix, full: b1379(ix, set() if full else {"Y_sd5", "X_n5d"}) + 0.05 * K11 * CLS + 0.10 * K11 * CLS * (slot == 1430) + 0.075 * K11 * CP,
}
for nm, f in DIRS.items():
    line, st = "%s 完整 − 平台" % nm, 0
    for wn, ix in W.items():
        rp = G.ev(np.where(z, 0.0, f(ix, False)), wn)
        rf = G.ev(np.where(z, 0.0, f(ix, True)), wn)
        d = {m: rf[m] - rp[m] for m in MET}
        st += ok(d)
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok(d) else "")
    print(line + "  %d/5" % st, flush=True)
