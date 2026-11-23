# -*- coding: utf-8 -*-
"""第十六轮 K3：除 14:30 之外，其他时刻有没有"结构性特殊段"。

对照 M2 的处理方式（锚 = 该段的过去 20 日 lab 均值，全天小权重 + 在对应截面再加一份），
逐槽位 IC 表（k1_anchor --mode slots）里除 14:30 外还有两个候选：
  - 11:30 截面：标签窗口跨午休（13:00–13:30 段），对应锚 = lab["1330"] 的 20 日均值；
  - 10:00 截面：早盘第一段的类比（锚 = lab["1030"] 的 20 日均值）。
只做"有经济理由"的两条，不加自由分槽位权重（G3 的教训）。
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
from h1_season import roll_prev  # noqa: E402

CANDS = [("午休前11:30(锚=13:30段)", "1330", 1130),
         ("早盘10:00(锚=10:30段)", "1030", 1000)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="1398,1399")
    ap.add_argument("--submit", default="")
    ap.add_argument("--submit-cand", default="午休前11:30(锚=13:30段)")
    a = ap.parse_args()
    G = GO()
    c = G.c
    kd = pd.to_datetime(c.key_date)
    z930 = c.slot_of_date[c.date_id] == 930
    slot = c.slot_of_date[c.date_id]
    D = K.load_frames()
    lab = D["lab"]
    AN = {nm: roll_prev(lab[seg], 20, 10) for nm, seg, _ in CANDS}

    BASES = {}
    if "1398" in a.bases:
        BASES["1398"] = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    if "1399" in a.bases:
        BASES["1399"] = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(B, float)), start=st, end=_en(en))
        return m

    def compose(B, ix, wn, r, bidx, boost):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        sd = K.xsec_sd(c, B[ix], st, en)
        aa = K.A_RANK * sd / K.RANK_SD
        comp = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan) + aa * r
        if boost:
            comp = comp + aa * r * (slot == bidx)
        return np.where(z930, 0.0, comp)

    RS = {}
    for nm, seg, bidx in CANDS:
        RS[nm] = K.rank_of(c, AN[nm], z930)[1]
    for bn, B in BASES.items():
        print("\n=== 底座 %s ===" % bn, flush=True)
        base_met = {wn: ev(B[ix], wn) for wn, ix in K.WINMAP}
        print("  底座 " + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
            wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in K.WINMAP), flush=True)
        for nm, seg, bidx in CANDS:
            for boost in (True, False):
                cells, okall = [], True
                for wn, ix in K.WINMAP:
                    m = ev(compose(B, ix, wn, RS[nm], bidx, boost), wn)
                    b = base_met[wn]
                    ok = (m["ic_mean"] >= b["ic_mean"] and m["ic_ir"] >= b["ic_ir"]
                          and m["stress"] >= b["stress"] and m["turnover"] <= b["turnover"] + 1e-9)
                    okall &= ok
                    cells.append("%s %+.5f/%+.4f/%+.4f%s" % (wn, m["ic_mean"] - b["ic_mean"],
                                                             m["ic_ir"] - b["ic_ir"],
                                                             m["turnover"] - b["turnover"], "★" if ok else ""))
                print("  %-24s%s %s -> %s" % (nm, "+加份" if boost else "仅全天", " ".join(cells), "★" if okall else "×"), flush=True)
    if a.submit:
        nm = a.submit_cand
        bidx = [x[2] for x in CANDS if x[0] == nm][0]
        B = BASES["1398"] if "1398" in BASES else None
        Bs = {}
        for wn, ix in (("main", 0), ("o21", 1), ("o20", 2)):
            Bs[wn] = compose(B, ix, wn, RS[nm], bidx, True)
        G.write_submit(a.submit, Bs, True)


if __name__ == "__main__":
    main()
