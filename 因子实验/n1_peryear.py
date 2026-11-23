# -*- coding: utf-8 -*-
"""N1：1363 + 修复版两根 的逐年增量（用缓存轴，秒级）。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g_go import GO  # noqa: E402

C = pathlib.Path("/tmp/n1_cache")


def main():
    G = GO()
    axY = np.load(C / "Y_2023-01.npy")
    axX = np.load(C / "X_2023-01.npy")
    base = np.asarray(G.B[("1363", 0)], float)
    fin = np.isfinite(base)
    fix = np.where(fin, base + 0.10 * np.nan_to_num(axY, nan=0.0) - 0.10 * np.nan_to_num(axX, nan=0.0), np.nan)
    wins = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
            ("main23-24", "2023-01-01", "2024-12-31")]
    for tag, B in (("1363", base), ("1363_fix", fix)):
        for nm, st, en in wins:
            m = G.evspan(B, st, en)
            print("%-10s %-10s IC %.5f IR %.4f 多空 %.2f 压力 %.4f 换手 %.4f"
                  % (tag, nm, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"]))


if __name__ == "__main__":
    main()
