# -*- coding: utf-8 -*-
"""第十七轮 L3：1398 + CLOSEPOS20（全天小权重）+ CLS20（只全天、不加 14:30）。

权重一律按秩口径 a = a0·σ_底座/0.289 换算（σ = 该窗口底座原始值的截面标准差中位数）。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
import k1_anchor as K  # noqa: E402
from g_go import GO, WINS, _en  # noqa: E402
from h1_stack import build as h1build, D1B, g4low_delta  # noqa: E402

CFG = [("CLS20全天0.05 + CP0.05", 0.05, 0.05, False),
       ("CLS20全天+14:30加份0.05 + CP0.05", 0.05, 0.05, True),
       ("CLS20全天0.10 + CP0.10", 0.10, 0.10, False),
       ("CLS20全天0.05 + CP0.10", 0.05, 0.10, False),
       ("只CP0.05（无季节性）", 0.0, 0.05, False),
       ("只CP0.10（无季节性）", 0.0, 0.10, False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="1398")
    ap.add_argument("--submit", default="")
    ap.add_argument("--submit-cfg", default=CFG[0][0])
    a = ap.parse_args()
    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    slot = c.slot_of_date[c.date_id]
    D = K.load_frames()
    AN = K.make_anchors(D)
    _, rcls = K.rank_of(c, AN["CLS20"], z930)
    _, rcp = K.rank_of(c, AN["CLOSEPOS20"], z930)
    B = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}

    def ev(Bx, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(Bx, float)), start=st, end=_en(en))
        return m

    SPAN = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
            "o20": ("2020-01-01", "2020-12-31")}

    def compose(ix, wn, ac0, ap0, boost):
        q = [x for x in WINS if x[0] == wn]
        st, en = (q[0][1], q[0][2]) if q else SPAN[wn]
        sd = K.xsec_sd(c, B[ix], st, en)
        ac = ac0 * sd / K.RANK_SD
        ap_ = ap0 * sd / K.RANK_SD
        x = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan)
        y = x + ac * rcls + ap_ * rcp
        if boost:
            y = y + ac * rcls * (slot == 1430)
        return np.where(z930, 0.0, y)

    print("=== 底座 %s ===" % a.base, flush=True)
    base_met = {wn: ev(B[ix], wn) for wn, ix in K.WINMAP}
    print("  底座 " + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
        wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in K.WINMAP), flush=True)
    for nm, ac0, ap0, boost in CFG:
        cells, okall = [], True
        for wn, ix in K.WINMAP:
            m = ev(compose(ix, wn, ac0, ap0, boost), wn)
            b = base_met[wn]
            ok = (m["ic_mean"] >= b["ic_mean"] and m["ic_ir"] >= b["ic_ir"]
                  and m["stress"] >= b["stress"] and m["turnover"] <= b["turnover"] + 1e-9)
            okall &= ok
            cells.append("%s %+.5f/%+.4f/%+.4f%s" % (wn, m["ic_mean"] - b["ic_mean"],
                                                     m["ic_ir"] - b["ic_ir"],
                                                     m["turnover"] - b["turnover"], "★" if ok else ""))
        print("  %-32s %s -> %s" % (nm, " ".join(cells), "★" if okall else "×"), flush=True)
    if a.submit:
        cfg = [x for x in CFG if x[0] == a.submit_cfg][0]
        Bs = {}
        for wn, ix in (("main", 0), ("o21", 1), ("o20", 2)):
            Bs[wn] = compose(ix, wn, cfg[1], cfg[2], cfg[3])
        G.write_submit(a.submit, Bs, True)


if __name__ == "__main__":
    main()
