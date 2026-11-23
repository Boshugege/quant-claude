# -*- coding: utf-8 -*-
"""分截面诊断：给定因子在各截面的 IC（用于判断 09:30 等截面是否反向）。"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '工具', '本地评分'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import localeval as le
import factor_lib as fl
from lab import get_panel


def per_slot(p, f, name):
    r = le.style_residual(f, p)
    m = p["is_eval"].values & np.isfinite(r.values)
    d = pd.DataFrame({"slot": p["slot"].values[m], "date": p["date"].values[m],
                      "f": r.values[m], "y": p["y_off"].values[m]})
    d = d[np.isfinite(d["y"])]
    print("== %s" % name)
    for s, g in d.groupby("slot"):
        ic = g.groupby("date").apply(lambda x: x["f"].corr(x["y"], method="spearman") if len(x) > 50 else np.nan).dropna()
        print("   %6d  截面 %4d  IC %+.4f  IC_IR %+.3f  正比例 %.2f" % (s, len(ic), ic.mean(), ic.mean() / max(ic.std(), 1e-9), (ic > 0).mean()))
    return d


if __name__ == "__main__":
    p = get_panel("2023-01", "2023-12")
    per_slot(p, fl.neutral4(p, fl.peer_res_ladder(p), 1.5), "104 同伴残差1.5倍中性")
    per_slot(p, fl.ladder(p, ["30", "rv4", "rv8", "day"], False), "90 四尺度阶梯")
    per_slot(p, -fl.ladder(p, ["30", "rv4", "rv8", "day"], False), "90 阶梯取负（动量方向）")
