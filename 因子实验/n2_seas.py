# -*- coding: utf-8 -*-
"""第十八轮 N2：季节性根（L_lab10 = 同时段季节性）权重下调的剂量实验。

背景：公榜回分标定出"同时段季节性根（1350＝1232 单独版）公榜 IC 只有本地的 0.65 倍"，
而它在 1363 里权重最大（0.20）。本脚本直接在平台实测的 1363 向量上按 δ 扣减季节性根的秩：
    B(δ) = B_1363 − δ · rank(L_lab10)
（合成式 B_1363 = 490 头部原始值 + Σ w·rank(根)，L_lab10 的 w = 0.20，见 因子代码/1363_*.py）
再给"公榜兑现率调整后"的 IC：
    IC_pub_adj(w) = 1.18 · f(0) + 0.7475 · (f(w) − f(0))     （f = 主区间 IC，w = 0.20 − δ）
其中 1.18 = 非季节性根的 IC 放大倍率（490／152／1353 标定），0.7475 = 0.65 × 1.15（季节性根）。
"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g_go import GO  # noqa: E402

DELTAS = [0.0, 0.02, 0.035, 0.05, 0.07, 0.10, 0.20]
LAB = "L_lab10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lab", default=LAB)
    a = ap.parse_args()
    G = GO()
    print("轴：%s" % a.lab)
    out = {}
    for ix, tag in ((0, "main(2023-24)"), (1, "2021-2022"), (2, "2020")):
        base = np.asarray(G.B[("1363", ix)], float)
        axL = G.axis(a.lab, ix)
        fin = np.isfinite(base)
        rows = []
        for d in DELTAS:
            B = base - d * np.nan_to_num(axL, nan=0.0)
            B = np.where(fin, B, np.nan)
            m = G.evspan(B, *{"main(2023-24)": ("2023-01-01", "2024-12-31"),
                              "2021-2022": ("2021-01-01", "2022-12-31"),
                              "2020": ("2020-01-01", "2020-12-31")}[tag])
            r0 = G.evspan(np.where(fin, base, np.nan), *{"main(2023-24)": ("2023-01-01", "2024-12-31"),
                                                          "2021-2022": ("2021-01-01", "2022-12-31"),
                                                          "2020": ("2020-01-01", "2020-12-31")}[tag])
            rows.append((d, m, r0))
        out[tag] = rows

    for tag, rows in out.items():
        print("\n=== %s ===" % tag)
        print("%-7s %-7s %9s %9s %9s %9s %9s | %s" % ("δ", "w", "IC", "IC_IR", "多空", "压力", "换手", "Δ vs 1363"))
        for d, m, r in rows:
            print("%-7.3f %-7.3f %9.5f %9.4f %9.2f %9.4f %9.4f | IC %+.5f IR %+.4f LS %+.2f 压力 %+.4f 换手 %+.4f"
                  % (d, 0.20 - d, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"],
                     m["ic_mean"] - r["ic_mean"], m["ic_ir"] - r["ic_ir"], m["ls_sharpe"] - r["ls_sharpe"],
                     m["stress"] - r["stress"], m["turnover"] - r["turnover"]))

    # 公榜兑现率调整
    rows = out["main(2023-24)"]
    f0 = rows[-1][1]["ic_mean"]                        # δ=0.20 → 季节性根完全去掉
    print("\n=== 主区间：公榜兑现率调整后 IC（f0 = 去季节性 %.5f） ===" % f0)
    print("%-7s %-7s %9s %9s %9s %9s" % ("δ", "w", "IC_local", "IC_seas", "IC_pub_adj", "换手"))
    for d, m, r in rows:
        ic = m["ic_mean"]
        print("%-7.3f %-7.3f %9.5f %9.5f %9.5f %9.4f"
              % (d, 0.20 - d, ic, ic - f0, 1.18 * f0 + 0.7475 * (ic - f0), m["turnover"]))


if __name__ == "__main__":
    main()
