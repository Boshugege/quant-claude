# -*- coding: utf-8 -*-
"""示例：评估自己写的候选因子（开发 / 验证两段，与固定参照池比较）。

用法（在本目录执行）：
    python3 example_custom_factor.py            # 单进程
    python3 example_custom_factor.py --jobs 8   # 多进程并行（Linux / macOS，fork 共享面板内存）

候选函数接收面板 p，返回与 p 行对齐的 Series（越大越看多）。可用列见 README“面板字段”。
"""
import argparse
import warnings

import numpy as np
import pandas as pd

import research as rs
from factor_lib import grolling, rank_blend

warnings.filterwarnings("ignore")


def my_vol_scaled_reversal(p):
    """示例 1：日内反转除以个股近 40 个窗口波动。"""
    return (p["r_day"] / p["sigma"]).fillna(p["r_day"])


def my_multi_scale(p):
    """示例 2：三个尺度的反转截面排名等权。"""
    return rank_blend(p, ["r_30", "r_dvwap", "r_day"])


def my_low_vol(p):
    """示例 3：低波动（近 40 个窗口收益标准差取负）。"""
    r = np.log(p["px"] / p.groupby("instrument_id", sort=False)["px"].shift(1))
    return -grolling(r, p["instrument_id"], 40, "std", min_periods=10)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--splits", nargs="*", default=["开发", "验证"], help="可选 开发 / 验证 / 保留（保留集建议最后只用一次）")
    args = ap.parse_args()
    cands = {"波动标准化日内反转": my_vol_scaled_reversal, "三尺度反转": my_multi_scale, "低波动": my_low_vol}
    res, long = rs.evaluate_candidates(cands, splits=tuple(args.splits), n_jobs=args.jobs)
    pd.set_option("display.width", 250)
    print(long[["候选", "切分", "ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]].round(4).to_string(index=False))
    print("\n综合分（候选单独放入固定参照池后的五项百分位均值）：\n" + res.round(4).to_string())
