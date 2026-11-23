# -*- coding: utf-8 -*-
"""第十六轮 K2 组合：尾盘季节性锚（K1 的 CLS20，14:30 再加一份）＋ CLOSEPOS20 锚（全天小权重）。

K1 的锚负责"提分"，K2 的 CLOSEPOS20 负责"降换手"，两者合起来看能不能同时拿到。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import k1_anchor as K  # noqa: E402
from g_go import GO, WINS, _en  # noqa: E402
from h1_stack import build as h1build, D1B, g4low_delta  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="1399")
    ap.add_argument("--close-a", type=float, default=0.05)
    ap.add_argument("--cp-doses", default="0.05,0.10")
    a = ap.parse_args()
    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    slot = c.slot_of_date[c.date_id]
    D = K.load_frames()
    AN = K.make_anchors(D)
    _, rcls = K.rank_of(c, AN["CLS20"], z930)
    _, rcp = K.rank_of(c, AN["CLOSEPOS20"], z930)

    BASES = {}
    if "1398" in a.bases:
        BASES["1398"] = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    if "1399" in a.bases:
        BASES["1399"] = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(B, float)), start=st, end=_en(en))
        return m

    for bn, B in BASES.items():
        print("\n=== 底座 %s ===" % bn, flush=True)
        base_met = {wn: ev(B[ix], wn) for wn, ix in K.WINMAP}
        print("  底座 " + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
            wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in K.WINMAP), flush=True)
        for cpd in [float(v) for v in a.cp_doses.split(",")]:
            cells, okall = [], True
            for wn, ix in K.WINMAP:
                st, en = [q for q in WINS if q[0] == wn][0][1:3]
                sd = K.xsec_sd(c, B[ix], st, en)
                ac = a.close_a * sd / K.RANK_SD   # 秩口径权重直接给
                ap_ = cpd * sd / K.RANK_SD
                comp = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan)
                comp = comp + ac * rcls + ac * rcls * (slot == 1430) + ap_ * rcp
                m = ev(np.where(z930, 0.0, comp), wn)
                b = base_met[wn]
                ok = (m["ic_mean"] >= b["ic_mean"] and m["ic_ir"] >= b["ic_ir"]
                      and m["stress"] >= b["stress"] and m["turnover"] <= b["turnover"] + 1e-9)
                okall &= ok
                cells.append("%s %+.5f/%+.4f/%+.4f%s" % (wn, m["ic_mean"] - b["ic_mean"],
                                                         m["ic_ir"] - b["ic_ir"],
                                                         m["turnover"] - b["turnover"], "★" if ok else ""))
            print("  CLS20(%.2f)+CLOSEPOS20(%.2f) %s -> %s" % (a.close_a, cpd, " ".join(cells), "★" if okall else "×"), flush=True)


if __name__ == "__main__":
    main()
