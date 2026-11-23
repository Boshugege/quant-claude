# -*- coding: utf-8 -*-
"""本地评分入口。

用法（在项目根目录执行）：
    python3 工具/本地评分/run.py                          # 默认 2023-01~2024-12，评估全部因子
    python3 工具/本地评分/run.py --start 2024-01 --end 2024-12 --only 69 80 81

输出：终端表格 + 工具/本地评分/结果/ 下的 CSV 与 Markdown。
"""
import argparse
import warnings
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import localeval as le  # noqa: E402
from factor_lib import FACTORS, add_components  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果")


def score_all(panel, factors, label):
    rows = []
    for fid, name, plat, fn, note in factors:
        m = le.evaluate(fn(panel), panel, label=label)
        rows.append({"编号": fid, "因子": name, "公榜分数": plat, "备注": note,
                     "ic_mean": m["ic_mean"], "ic_ir": m["ic_ir"], "ls_sharpe": m["ls_sharpe"],
                     "stress": m["stress"], "turnover": m["turnover"]})
    t = pd.DataFrame(rows).set_index("编号")
    t["本地综合分"] = le.composite(t)
    return t.sort_values("本地综合分", ascending=False)


def calibration(t):
    k = t.dropna(subset=["公榜分数"])
    rho = k["本地综合分"].rank().corr(k["公榜分数"].rank())
    rho_ic = k["ic_mean"].rank().corr(k["公榜分数"].rank())
    return rho, rho_ic, len(k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01")
    ap.add_argument("--end", default="2024-12")
    ap.add_argument("--only", nargs="*", help="只评估这些编号（仍与全部已知因子一起排名）")
    args = ap.parse_args()

    t0 = time.time()
    panel = le.build_panel(le.load_bars(args.start, args.end))
    panel = add_components(panel)
    print(f"数据 {args.start}~{args.end}：{len(panel):,} 行，{panel['instrument_id'].nunique()} 只股票，"
          f"{panel['day'].nunique()} 个交易日，准备耗时 {time.time() - t0:.0f}s")

    factors = FACTORS
    if args.only:
        factors = [f for f in FACTORS if f[0] in set(args.only) or f[2] is not None]

    labels = ["y_off"]   # 官方标签口径（e2e/reports/12）
    os.makedirs(OUT_DIR, exist_ok=True)
    for label in labels:
        t = score_all(panel, factors, label)
        rho, rho_ic, n = calibration(t)
        print(f"\n===== 标签 {label}：本地综合分与公榜分数的秩相关 = {rho:.3f}（仅 IC 均值 = {rho_ic:.3f}，n={n}）=====")
        show = t.copy()
        for c in ["ic_mean", "stress"]:
            show[c] = show[c].map(lambda v: f"{v:+.4f}")
        for c in ["ic_ir", "ls_sharpe"]:
            show[c] = show[c].map(lambda v: f"{v:+.2f}")
        show["turnover"] = show["turnover"].map(lambda v: f"{v:.3f}")
        show["本地综合分"] = show["本地综合分"].map(lambda v: f"{v:.3f}")
        show["公榜分数"] = show["公榜分数"].map(lambda v: "" if pd.isna(v) else f"{v:.5f}")
        print(show.to_string())
        stem = f"本地评分_{args.start}_{args.end}_{label}"
        t.to_csv(os.path.join(OUT_DIR, stem + ".csv"), encoding="utf-8-sig")
        with open(os.path.join(OUT_DIR, stem + ".md"), "w", encoding="utf-8") as fh:
            fh.write(f"# 本地评分 {args.start}~{args.end}，标签 {label}\n\n")
            fh.write(f"本地综合分与公榜分数秩相关 {rho:.3f}（仅 IC 均值 {rho_ic:.3f}，n={n}）\n\n")
            cols = ["编号"] + list(show.columns)
            fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
            for fid, row in show.iterrows():
                fh.write("| " + " | ".join([str(fid)] + [str(v) for v in row.values]) + " |\n")
    print(f"\n总耗时 {time.time() - t0:.0f}s；结果已写入 {OUT_DIR}")


if __name__ == "__main__":
    main()
