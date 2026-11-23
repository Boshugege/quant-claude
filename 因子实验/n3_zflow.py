# -*- coding: utf-8 -*-
"""第十八轮 N3：把资金流族的新变体叠到 1363 上（主区间 2023–24），判断有没有占名额的潜力。"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
from g_go import GO  # noqa: E402

LABS = ["fm_18b_Z_mfr5d", "fm_18b_Z_bigup", "fm_18b_Z_flowgap", "fm_18_Z_mfr5d"]
WS = [-0.05, -0.10, -0.15, -0.20, -0.25]


def main():
    G = GO()
    c = G.c
    base = np.asarray(G.B[("1363", 0)], float)
    fin = np.isfinite(base)
    st, en = "2023-01-01", "2024-12-31"
    r0 = G.evspan(np.where(fin, base, np.nan), st, en)
    print("底座 1363：IC %.5f IR %.4f 多空 %.2f 压力 %.4f 换手 %.4f"
          % (r0["ic_mean"], r0["ic_ir"], r0["ls_sharpe"], r0["stress"], r0["turnover"]))
    for lab in LABS:
        try:
            ax = O.axis(c, lab, st, en)
        except SystemExit as e:
            print("%s 缺（%s）" % (lab, e))
            continue
        ax = np.nan_to_num(ax, nan=0.0)
        print("\n=== %s ===" % lab)
        for w in WS:
            B = np.where(fin, base + w * ax, np.nan)
            m = G.evspan(B, st, en)
            print("  w %+.2f | IC %.5f (%+.5f) IR %.4f (%+.4f) 多空 %.2f (%+.2f) 压力 %.4f (%+.4f) 换手 %.4f (%+.4f)"
                  % (w, m["ic_mean"], m["ic_mean"] - r0["ic_mean"], m["ic_ir"], m["ic_ir"] - r0["ic_ir"],
                     m["ls_sharpe"], m["ls_sharpe"] - r0["ls_sharpe"], m["stress"], m["stress"] - r0["stress"],
                     m["turnover"], m["turnover"] - r0["turnover"]))


if __name__ == "__main__":
    main()
