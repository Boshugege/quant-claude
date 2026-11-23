# -*- coding: utf-8 -*-
"""第七批公榜回分对照：各文件的本地五项（2023、2024，engine 口径）——"已交版"（含成交笔数根）与"平台一致版"（成交笔数根置零）。"""
import numpy as np
import root_eval as R
from c1_upgrades import G, build, MET
z = G.z930
def ev(B):
    B = np.asarray(B, float).copy(); B[z] = 0.0
    return {wn: G.ev(B, wn) for wn in ("2023", "2024")}
rows = {
    "1363": G.B[("1363", 0)],
    "s152": R.load("fm_s152")[0],
    "s161": G.B[("s161", 0)],
    "1602已交": build("1602", 0, drop_deal=False),
    "1602平台": build("1602", 0),
    "1397": build("1601", 0, drop_deal=False, cls_all=0.0, cls_boost=0.0),
    "1601已交": build("1601", 0, drop_deal=False),
    "1601平台": build("1601", 0),
    "1399": build("1603", 0, drop_deal=False, cls_all=0.0, cls_boost=0.0),
    "1603已交": build("1603", 0, drop_deal=False),
    "1603平台": build("1603", 0),
}
for k, B in rows.items():
    r = ev(B)
    print("%-9s " % k + " | ".join("%s %.4f/%.4f/%.2f/%.4f/%.4f" % (wn, *[r[wn][m] for m in MET]) for wn in ("2023", "2024")), flush=True)
