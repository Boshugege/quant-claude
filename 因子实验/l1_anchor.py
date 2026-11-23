# -*- coding: utf-8 -*-
"""第十七轮 L1：把"收盘行为族"组一个锚。

用法：--mode select 先按 2020–2022 的单锚 IC 选特征与权重（不 evaluate）；--mode eval 再在 1398 与
`v_CLS1399`（1399+CLS20，14:30 加份）两个底座上评估。

权重规则（只在 2020–22 定）：先按 2020–22 的单锚 IC 从大到小贪心挑选，与已选锚的相关 < 0.7；
权重 w_k ∝ IC_k(2020–22)，归一化到总秩权重 a（默认 0.05），合成 = 底座原始值 + Σ (w_k·σ_底座/0.289)·rank_k。
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

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
FEATS = ["LAB20", "POSHI20", "POSLO20", "RNG20", "R5_20", "R15_20", "R30_20",
         "AUCK20", "AUV20", "AMTSH30_20", "AMTSH5_20", "DEV20", "CLS20", "CLOSEPOS20"]
FIT_YEARS = (2020, 2021, 2022)


def load_ranks(G, feats):
    c = G.c
    z930 = c.slot_of_date[c.date_id] == 930
    Rk = {}
    for nm in feats:
        x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
        Rk[nm] = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
    return Rk, z930


def select(G, Rk, a_total, thr=0.002, rho=0.7):
    c = G.c
    kd = pd.to_datetime(c.key_date)
    y = np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)
    m = np.isin(kd.year, FIT_YEARS) & (c.slot_of_date[c.date_id] != 930)
    dm = np.zeros(c.n_dates, bool)
    dm[np.unique(c.date_id[m])] = True
    ic = {nm: c.per_date_rank_corr(r, y, dm) for nm, r in Rk.items()}
    order = sorted([n for n in ic if ic[n] > thr], key=lambda n: -ic[n])
    picked = []
    for nm in order:
        if all(abs(c.per_date_rank_corr(Rk[nm], Rk[p], dm)) < rho for p in picked):
            picked.append(nm)
    tot = sum(ic[n] for n in picked)
    w = {n: a_total * ic[n] / tot for n in picked}
    print("2020–22 单锚 IC：" + " ".join("%s %+.4f" % (n, ic[n]) for n in sorted(ic, key=lambda z: -ic[z])), flush=True)
    print("入选（相关<%.1f）：%s" % (rho, " ".join("%s %.4f" % (n, w[n]) for n in picked)), flush=True)
    return w, ic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="eval")
    ap.add_argument("--bases", default="1398,CLS1399")
    ap.add_argument("--doses", default="0.05,0.10")
    ap.add_argument("--feats", default=",".join(FEATS))
    ap.add_argument("--submit", default="")
    ap.add_argument("--submit-base", default="CLS1399")
    ap.add_argument("--submit-dose", type=float, default=0.05)
    a = ap.parse_args()
    G = GO()
    c = G.c
    slot = c.slot_of_date[c.date_id]
    feats = [f for f in a.feats.split(",") if f]
    Rk, z930 = load_ranks(G, feats)
    w, ic = select(G, Rk, float(a.doses.split(",")[0]))

    Bs = {}
    if "1398" in a.bases:
        Bs["1398"] = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    if "CLS1399" in a.bases:
        D = K.load_frames()
        AN = K.make_anchors(D)
        _, rcls = K.rank_of(c, AN["CLS20"], z930)
        B9 = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}
        SPAN = {0: ("2023-01-01", "2024-12-31"), 1: ("2021-01-01", "2022-12-31"), 2: ("2020-01-01", "2020-12-31")}
        Bs["CLS1399"] = {}
        for ix in (0, 1, 2):
            sd = K.xsec_sd(c, B9[ix], *SPAN[ix])
            ac = 0.05 * sd / K.RANK_SD
            x = np.where(np.isfinite(B9[ix]), np.asarray(B9[ix], float), np.nan)
            Bs["CLS1399"][ix] = np.where(z930, 0.0, x + ac * rcls + ac * rcls * (slot == 1430))

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(B, float)), start=st, end=_en(en))
        return m

    SPAN = {wn: (st, en) for wn, st, en in WINS}

    def comp(ix, st, en, dose, names):
        base = Bs[bname][ix]
        sd = K.xsec_sd(c, base, st, en)
        x = np.where(np.isfinite(base), np.asarray(base, float), np.nan)
        tot = sum(w[n] for n in names) if names else 1.0
        for n in names:
            x = x + (dose * w[n] / tot) * sd / K.RANK_SD * Rk[n]
        return np.where(z930, 0.0, x)

    order = [n for n in sorted(w, key=lambda z: -w[z])]
    for bname in a.bases.split(","):
        print("\n=== 底座 %s ===" % bname, flush=True)
        base_met = {wn: ev(Bs[bname][ix], wn) for wn, ix in K.WINMAP}
        print("  底座 " + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
            wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in K.WINMAP), flush=True)
        cfgs = []
        for d in [float(v) for v in a.doses.split(",")]:
            cfgs.append(("全族 a=%.2f" % d, d, order))
        cfgs.append(("单锚 %s a=0.05" % order[0], 0.05, [order[0]]))
        if len(order) > 1:
            cfgs.append(("双锚 %s+%s a=0.05" % (order[0], order[1]), 0.05, order[:2]))
        for nm, d, names in cfgs:
            cells, okall = [], True
            for wn, ix in K.WINMAP:
                m = ev(comp(ix, *SPAN[wn], d, names), wn)
                b = base_met[wn]
                ok = (m["ic_mean"] >= b["ic_mean"] and m["ic_ir"] >= b["ic_ir"]
                      and m["stress"] >= b["stress"] and m["turnover"] <= b["turnover"] + 1e-9)
                okall &= ok
                cells.append("%s %+.5f/%+.4f/%+.4f%s" % (wn, m["ic_mean"] - b["ic_mean"],
                                                         m["ic_ir"] - b["ic_ir"],
                                                         m["turnover"] - b["turnover"], "★" if ok else ""))
            print("  %-26s %s -> %s" % (nm, " ".join(cells), "★" if okall else "×"), flush=True)
    if a.submit:
        SUBSPAN = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
                   "o20": ("2020-01-01", "2020-12-31")}
        opts = list(zip([x for x in a.submit.split(",") if x], [x for x in a.submit_base.split(",") if x]))
        for nm, bn in opts:
            bname = bn
            Bs2 = {}
            for wn, ix in (("main", 0), ("o21", 1), ("o20", 2)):
                Bs2[wn] = comp(ix, *SUBSPAN[wn], a.submit_dose, order)
            G.write_submit(nm, Bs2, True)


if __name__ == "__main__":
    main()
