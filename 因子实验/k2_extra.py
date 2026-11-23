# -*- coding: utf-8 -*-
"""第十六轮 K2 补充：唯一有效的锚 CLOSEPOS20 的剂量扫描与冗余度。

CLOSEPOS20 = (收盘价 ÷ 当日最高价 − 1) 的过去 20 日均值（当日不变、只用严格过去交易日）。
剂量 = 全天小权重 a（秩口径 0.05 为 M2 的基准）。
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="1398,1399")
    ap.add_argument("--doses", default="0.05,0.10,0.20")
    a = ap.parse_args()
    G = GO()
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    D = K.load_frames()
    AN = K.make_anchors(D)
    x, r = K.rank_of(c, AN["CLOSEPOS20"], z930)
    _, rc = K.rank_of(c, AN["CLS20"], z930)

    BASES = {}
    if "1398" in a.bases:
        BASES["1398"] = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    if "1399" in a.bases:
        BASES["1399"] = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(B, float)), start=st, end=_en(en))
        return m

    # 冗余度：锚与底座、与 CLS20、与 s161 头部的逐截面秩相关（区间均值）
    print("=== 冗余度（逐截面秩相关均值）===", flush=True)
    for wn, ix in (("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2)):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        dm = K.date_mask_of(c, st, en)
        print("  %s  CLOSEPOS20 vs 1398 %+.4f | vs CLS20 %+.4f" % (
            wn, c.per_date_rank_corr(np.nan_to_num(x, nan=0.0), G.B[("s161", ix)], dm),
            c.per_date_rank_corr(np.nan_to_num(x, nan=0.0), np.nan_to_num(rc, nan=0.0), dm)), flush=True)

    doses = [float(v) for v in a.doses.split(",")]
    for bn, B in BASES.items():
        print("\n=== 底座 %s ===" % bn, flush=True)
        base_met = {wn: ev(B[ix], wn) for wn, ix in K.WINMAP}
        print("  底座 " + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
            wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in K.WINMAP), flush=True)
        for tot in (doses if bn == "1398" else doses[::2]):
            cells, okall = [], True
            for wn, ix in K.WINMAP:
                st, en = [q for q in WINS if q[0] == wn][0][1:3]
                sd = K.xsec_sd(c, B[ix], st, en)
                aa = tot * sd / K.RANK_SD        # tot 即秩口径权重 a，不再乘 A_RANK
                comp = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan) + aa * r
                m = ev(np.where(z930, 0.0, comp), wn)
                b = base_met[wn]
                ok = (m["ic_mean"] >= b["ic_mean"] and m["ic_ir"] >= b["ic_ir"]
                      and m["stress"] >= b["stress"] and m["turnover"] <= b["turnover"] + 1e-9)
                okall &= ok
                cells.append("%s %+.5f/%+.4f/%+.4f%s" % (wn, m["ic_mean"] - b["ic_mean"],
                                                         m["ic_ir"] - b["ic_ir"],
                                                         m["turnover"] - b["turnover"], "★" if ok else ""))
            print("  a=%.2f %s -> %s" % (tot, " ".join(cells), "★" if okall else "×"), flush=True)


if __name__ == "__main__":
    main()
