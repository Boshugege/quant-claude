# -*- coding: utf-8 -*-
"""第十四轮 H1（后半）：把季节性族叠到 1398（G1B_B1）与 1399（G4LOW）上，五个窗口判 ★。

★ 判定（与 g_go 一致）：IC/IC_IR/多空/压力 不显著变差、换手不升。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402
from g_go import GO, WINS, IXW, MET, TOL, _en  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
HEADTASK = {"s161": ["fm_s161", "o21_s161", "o20_s161"], "s155": ["fm_s155", "o21_s155", "o20_s155"],
            "s152": ["fm_s152", "o21_s152", "o20_s152"], "1363": ["fm_1363", "o21_1363", "o20_1363"]}
D1B = {"L_lab10": 0.20, "M_q_lin": 0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": 0.10,
       "Y_sd5": 0.10, "X_n5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10,
       "k1381": 0.02, "k1385": 0.02, "k1406": 0.05}
BASES = {"G1B_B1": ("s161", D1B),
         "G4LOW": ("1363", None)}
WINMAP = [("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2)]


def build(G, ix, head, delta):
    c = G.c
    if head == "1363":
        B = np.asarray(G.B[("1363", ix)], float).copy()
    else:
        rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
        B = np.nan_to_num(O.axis(c, HEADTASK[head][ix], *rng), nan=0.0)
    m = np.isfinite(B)
    for lab, v in (delta or {}).items():
        if abs(v) < 1e-13:
            continue
        if lab in HEADTASK:
            rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
            B = B + v * np.nan_to_num(O.axis(c, HEADTASK[lab][ix], *rng), nan=0.0)
        else:
            B = B + v * G.axis(lab, ix)
    return np.where(m, B, np.nan)


def g4low_delta():
    d = json.load(open(OUT / "g4_opt.json"))["runs"]["3.0000"]["delta"]
    return {k: float(v) for k, v in d.items() if abs(float(v)) > 1e-13}


def stev(G, B, wn):
    st, en = [q for q in WINS if q[0] == wn][0][1:3]
    m, _ = G.c.evaluate(np.where(G.z930, 0.0, B), start=st, end=_en(en))
    return m


def main():
    global WINMAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", default="")
    ap.add_argument("--ws", default="0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--bases", default="G1B_B1,G4LOW")
    ap.add_argument("--blend", default="")
    ap.add_argument("--wins", default="")
    ap.add_argument("--detail", action="store_true")
    a = ap.parse_args()
    G = GO()
    c = G.c
    names = a.members.split(",") if a.members else sorted(
        p.name[3:-4] for p in OUT.glob("h1_*.npy"))
    fam = {}
    for nm in names:
        x = np.load(OUT / ("h1_%s.npy" % nm))
        fam[nm] = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
    print("族成员 %s | 窗口 %s" % (list(fam), [q[0] for q in WINMAP]))
    ws = [float(x) for x in a.ws.split(",")]
    if a.wins:
        keep = set(a.wins.split(","))
        WINMAP = [q for q in WINMAP if q[0] in keep]
    d4 = g4low_delta()
    for bname in a.bases.split(","):
        head, delta = BASES[bname]
        if bname == "G4LOW":
            delta = d4
        B = {ix: build(G, ix, head, delta) for ix in (0, 1, 2)}
        ref = {wn: stev(G, B[ix], wn) for wn, ix in WINMAP}
        print("\n===== 底座 %s =====" % bname)
        print("  底座：" + " ".join("%s IC %.5f IR %.4f 压力 %.4f 换手 %.4f" %
                                   (wn, ref[wn]["ic_mean"], ref[wn]["ic_ir"], ref[wn]["stress"], ref[wn]["turnover"])
                                   for wn, _ in WINMAP))
        for nm in list(fam) + (["BLEND"] if a.blend else []):
            if nm == "BLEND":
                sel = a.blend.split(",")
                sig = sum(fam[s] for s in sel) / len(sel)
            else:
                sig = fam[nm]
            for w in ws:
                cells, ok_all, det = [], True, []
                for wn, ix in WINMAP:
                    m = stev(G, B[ix] + w * sig, wn)
                    ok = all((m[k] - ref[wn][k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in MET)
                    ok_all &= ok
                    cells.append("%+.4f%s" % (m["ic_mean"] - ref[wn]["ic_mean"], "★" if ok else " "))
                    det.append("%s IC %+.5f IR %+.4f LS %+.2f 压力 %+.4f 换手 %+.4f" %
                               (wn, m["ic_mean"] - ref[wn]["ic_mean"], m["ic_ir"] - ref[wn]["ic_ir"],
                                m["ls_sharpe"] - ref[wn]["ls_sharpe"], m["stress"] - ref[wn]["stress"],
                                m["turnover"] - ref[wn]["turnover"]))
                print("  %-8s w=%+.2f %s  %s" % (nm, w, "".join("%-8s" % x for x in cells),
                                                 "全★" if ok_all else "   "))
                if a.detail or ok_all or any(x.strip().endswith("★") for x in cells):
                    print("       " + " | ".join(det))


if __name__ == "__main__":
    main()
