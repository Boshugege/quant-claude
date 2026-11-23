# -*- coding: utf-8 -*-
"""N1：按"五根底座 1360 → 加两根 1361"的口径复核增量（平台 1363 = 1360 + 资金流两列）。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
from g_go import GO  # noqa: E402

C = pathlib.Path("/tmp/n1_cache")


def main():
    G = GO()
    c = G.c
    base = np.asarray(G.B[("1363", 0)], float)
    fin = np.isfinite(base)
    ax60 = np.nan_to_num(O.axis(c, "fm_1334", "2023-01-01", "2024-12-31"), nan=0.0)
    ax90 = np.nan_to_num(O.axis(c, "fm_1342", "2023-01-01", "2024-12-31"), nan=0.0)
    axY = np.nan_to_num(np.load(C / "Y_2023-01.npy"), nan=0.0)
    axX = np.nan_to_num(np.load(C / "X_2023-01.npy"), nan=0.0)
    H1360 = np.where(fin, base + 0.15 * ax60 + 0.10 * ax90, np.nan)
    H1361 = np.where(fin, H1360 + 0.10 * axY - 0.10 * axX, np.nan)
    H1363 = np.where(fin, H1360 - 0.15 * ax60 - 0.10 * ax90 + 0.10 * axY - 0.10 * axX, np.nan)
    for nm, B in (("1360(五根)", H1360), ("1361(1360+两根固定)", H1361), ("1363(1360+资金流+两根固定)", H1363)):
        m = G.evspan(B, "2023-01-01", "2024-12-31")
        print("%-28s IC %.5f IR %.4f 多空 %.2f 压力 %.4f 换手 %.4f"
              % (nm, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"]))
    for nm, B in (("2023 1360→1361", None),):
        pass
    for nm, B in (("1360", H1360), ("1361", H1361)):
        for y, (st, en) in (("2023", ("2023-01-01", "2023-12-31")), ("2024", ("2024-01-01", "2024-12-31"))):
            m = G.evspan(B, st, en)
            print("  %-6s %s IC %.5f IR %.4f 换手 %.4f" % (nm, y, m["ic_mean"], m["ic_ir"], m["turnover"]))


if __name__ == "__main__":
    main()
