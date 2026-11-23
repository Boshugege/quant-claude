# -*- coding: utf-8 -*-
"""H1b：把"多回看季节性"做成 EWMA 族（α=0.75，与 1398/1399 里 L_lab10 的处理一致），
再叠到 1398（G1B_B1）与 1399（G4LOW）上判五窗口 ★。"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402
from g_go import GO, WINS, MET, TOL, _en  # noqa: E402
from h1_stack import build, BASES, g4low_delta  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
WINMAP = [("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2)]


def ewma_axis(c, raw, alpha=0.75):
    kd = np.asarray(c.key_date).astype("datetime64[ns]").astype(np.int64)
    inst = c.inst.astype(np.int64)
    d = pd.DataFrame({"inst": inst, "kd": kd, "v": np.nan_to_num(np.asarray(raw, float), nan=0.0)})
    d = d.sort_values(["inst", "kd"], kind="stable")
    d["e"] = d.groupby("inst", sort=False)["v"].transform(
        lambda x: x.ewm(alpha=1 - alpha, adjust=False, ignore_na=True).mean())
    e = np.empty(len(d))
    e[d.index.to_numpy()] = d["e"].to_numpy()
    return np.nan_to_num(O.rc_of(e, c.date_id, c.n_dates), nan=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bases", default="G1B_B1,G4LOW")
    ap.add_argument("--ws", default="0.15,0.25")
    ap.add_argument("--members", default="L_5,L_20,L_40")
    a = ap.parse_args()
    G = GO()
    c = G.c
    ms = a.members.split(",")
    raws = {m: np.load(OUT / ("h1_%s.npy" % m)) for m in ms}
    fam = np.nanmean(np.stack([raws[m] for m in ms]), axis=0)
    print("族平均：非零 %.3f 均值 %.5f" % ((fam != 0).mean(), np.nanmean(fam[fam != 0])))
    AX = {"FAM": ewma_axis(c, fam)}
    for m in ms:
        AX[m] = ewma_axis(c, raws[m])
    del raws
    d4 = g4low_delta()
    ws = [float(x) for x in a.ws.split(",")]

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(G.z930, 0.0, B), start=st, end=_en(en))
        return m

    for bname in a.bases.split(","):
        head, delta = BASES[bname]
        if bname == "G4LOW":
            delta = d4
        B = {ix: build(G, ix, head, delta) for ix in (0, 1, 2)}
        ref = {wn: ev(B[ix], wn) for wn, ix in WINMAP}
        print("\n===== 底座 %s =====" % bname)
        print("  底座：" + " ".join("%s IC %.5f IR %.4f 压力 %.4f 换手 %.4f" %
                                   (wn, ref[wn]["ic_mean"], ref[wn]["ic_ir"], ref[wn]["stress"], ref[wn]["turnover"])
                                   for wn, _ in WINMAP))
        for nm in ("FAM",) + tuple(ms):
            for w in ws:
                cells, ok_all, det = [], True, []
                for wn, ix in WINMAP:
                    m = ev(B[ix] + w * AX[nm], wn)
                    ok = all((m[k] - ref[wn][k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in MET)
                    ok_all &= ok
                    cells.append("%+.4f%s" % (m["ic_mean"] - ref[wn]["ic_mean"], "★" if ok else " "))
                    det.append("%s IC %+.5f IR %+.4f LS %+.2f 压力 %+.4f 换手 %+.4f" %
                               (wn, m["ic_mean"] - ref[wn]["ic_mean"], m["ic_ir"] - ref[wn]["ic_ir"],
                                m["ls_sharpe"] - ref[wn]["ls_sharpe"], m["stress"] - ref[wn]["stress"],
                                m["turnover"] - ref[wn]["turnover"]))
                print("  %-6s w=%+.2f %s  %s" % (nm, w, "".join("%-8s" % x for x in cells), "全★" if ok_all else ""))
                print("      " + " | ".join(det))


if __name__ == "__main__":
    main()
