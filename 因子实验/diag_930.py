# -*- coding: utf-8 -*-
"""09:30 截面专用诊断：逐一评估候选信号在该截面的 IC，寻找能替代现弱信号（0.045）的方向。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "工具", "本地评分"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import localeval as le
import factor_lib as fl
from lab import get_panel
from cand_r1 import auc_share, auc_share_hist, d5_rev, downside_share, vol_centroid
from cand_r2 import peer_res_ladder_w, _log_ratio


def slot_ic(p, f, slot=930):
    r = le.style_residual(f, p)
    m = p["is_eval"].values & (p["slot"].values == slot) & np.isfinite(r.values)
    d = pd.DataFrame({"date": p["date"].values[m], "f": r.values[m], "y": p["y_off"].values[m]})
    d = d[np.isfinite(d["y"])]
    ic = d.groupby("date").apply(lambda x: x["f"].corr(x["y"], method="spearman") if len(x) > 50 else np.nan).dropna()
    return ic.mean(), ic.mean() / max(ic.std(), 1e-9), len(ic)


if __name__ == "__main__":
    p = get_panel("2023-01", "2023-12")
    n = p["slot"].values == 930
    signals = {
        "104阶梯(现状)": fl.neutral4(p, fl.peer_res_ladder(p), 1.5),
        "r_day 隔夜反转": p["r_day"],
        "r_30 上一窗口反转": p["r_30"],
        "r_dvwap 日内VWAP偏离": p["r_dvwap"],
        "r_rv4 两小时VWAP偏离": fl.rv_raw(p, 4),
        "r_rrng 区间位置": p["r_rng"],
        "同伴残差r_day": fl.peer_res_ladder(p, parts=("day",)),
        "竞价额占比": auc_share(p),
        "竞价额占比偏离": auc_share_hist(p),
        "下行波动占比": downside_share(p),
        "5日反转": d5_rev(p),
        "成交重心偏离": vol_centroid(p),
        "开盘缺口反转": _log_ratio(p["pre_close"], p["open"]),
        "30分钟波动": p["sigma"],
    }
    print("== 09:30 截面（2023）逐信号 IC")
    for name, v in signals.items():
        ic, ir, k = slot_ic(p, v, 930)
        print("   %-22s IC %+.4f  IC_IR %+.3f  截面 %d" % (name, ic, ir, k))
