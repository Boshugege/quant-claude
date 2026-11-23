# -*- coding: utf-8 -*-
"""对已保存的 h2_<tag>_<year>.npy 做五窗口评估（engine.Core.evaluate），并与本地 1397 比较。"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
from g_go import GO, WINS, _en  # noqa: E402
from h2_robust import Ref1397  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="+")
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    ref = Ref1397()
    res = {}
    for tg in a.tags:
        row = {}
        for wn in ("2023", "2024", "2021", "2022", "2020"):
            v = np.load(OUT / ("h2_%s_%s.npy" % (tg, wn)))
            st, en = [q for q in WINS if q[0] == wn][0][1:3]
            m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(v, float)), start=st, end=_en(en))
            r = ref.get(c, wn)
            row[wn] = {k: float(m[k]) for k in MET}
            print("%-14s %s IC %.5f IR %.4f LS %.1f 压力 %.4f 换手 %.4f | ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f"
                  % (tg, wn, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"],
                     m["ic_mean"] - r["ic_mean"], m["ic_ir"] - r["ic_ir"],
                     m["stress"] - r["stress"], m["turnover"] - r["turnover"]), flush=True)
        res[tg] = row
        print("SUM %s %s" % (tg, json.dumps(row)), flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
