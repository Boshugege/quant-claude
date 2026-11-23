# -*- coding: utf-8 -*-
"""第十八轮 N3：把资金流族的新变体叠到任意底座（默认 v_CLSCP 的 main 任务）上，看有没有增量。"""
import argparse
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
import root_eval as R  # noqa: E402
from g_go import GO  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="1017-171349-c93c", help="底座任务名/任务号（默认 v_CLSCP main）")
    ap.add_argument("--labs", default="fm_18_Z_mfr5d,fm_18c_Z_cmfr60,fm_18c_Z_net5d")
    ap.add_argument("--ws", default="-0.05,-0.10,-0.15,-0.20")
    a = ap.parse_args()
    G = GO()
    c = G.c
    base = np.asarray(R.load(a.base)[0], float)
    base[c.slot_of_date[c.date_id] == 930] = 0.0
    fin = np.isfinite(base)
    st, en = "2023-01-01", "2024-12-31"
    r0 = G.evspan(np.where(fin, base, np.nan), st, en)
    print("底座 %s：IC %.5f IR %.4f 多空 %.2f 压力 %.4f 换手 %.4f"
          % (a.base, r0["ic_mean"], r0["ic_ir"], r0["ls_sharpe"], r0["stress"], r0["turnover"]))
    ws = [float(x) for x in a.ws.split(",")]
    for lab in [x.strip() for x in a.labs.split(",") if x.strip()]:
        try:
            ax = np.nan_to_num(O.axis(c, lab, st, en), nan=0.0)
        except SystemExit as e:
            print("\n%s 缺：%s" % (lab, e))
            continue
        print("\n=== %s ===" % lab)
        for w in ws:
            B = np.where(fin, base + w * ax, np.nan)
            m = G.evspan(B, st, en)
            print("  w %+.2f | IC %.5f (%+.5f) IR %.4f (%+.4f) 多空 %.2f (%+.2f) 压力 %.4f (%+.4f) 换手 %.4f (%+.4f)"
                  % (w, m["ic_mean"], m["ic_mean"] - r0["ic_mean"], m["ic_ir"], m["ic_ir"] - r0["ic_ir"],
                     m["ls_sharpe"], m["ls_sharpe"] - r0["ls_sharpe"], m["stress"], m["stress"] - r0["stress"],
                     m["turnover"], m["turnover"] - r0["turnover"]))


if __name__ == "__main__":
    main()
