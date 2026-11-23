# -*- coding: utf-8 -*-
"""Claude 夜间 C8：远端第十八轮留下的两个"形状安全"近 5 日根（不用成交笔数、纯 SUM/COUNT）能否在 1606 平台一致版上补回换手。
Y_z5（r_1755）、X_q5d（r_1754）只有主区间（2023–24）任务，先看这两年。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
from c1_upgrades import G, build, MET  # noqa: E402

c = G.c
ay = np.nan_to_num(O.axis(c, "r_1755", "2020-01-01", "2024-12-31"), nan=0.0)
ax_ = np.nan_to_num(O.axis(c, "r_1754", "2020-01-01", "2024-12-31"), nan=0.0)
ay[G.z930] = 0.0
ax_[G.z930] = 0.0
base = build("1603", 0, dose=1.5, cp=0.05)
rb = {wn: G.ev(base, wn) for wn in ("2023", "2024")}
for a in (0.0, 0.10, 0.15, -0.10, -0.15):
    for b in (0.0, 0.10, 0.15, -0.10, -0.15):
        if a == 0 and b == 0:
            continue
        B = base + a * ay + b * ax_
        line = "Y_z5 %+.2f X_q5d %+.2f" % (a, b)
        for wn in ("2023", "2024"):
            m = G.ev(B, wn)
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f" % (wn, *[m[k] - rb[wn][k] for k in MET])
        print(line, flush=True)
