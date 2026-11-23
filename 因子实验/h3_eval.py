# -*- coding: utf-8 -*-
"""第十四轮 H3：把 G1B_B1 带换手惩罚重定权（g_opt 的 λ=1/2/3/5 结果）在五个窗口判 ★。"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from g_go import GO, WINS, MET, TOL, _en  # noqa: E402
from h1_stack import build, D1B  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
WINMAP = [("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(OUT / "h3_opt.json"))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--lams", default="")
    a = ap.parse_args()
    j = json.load(open(a.json, encoding="utf-8"))
    G = GO()
    c = G.c

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(G.z930, 0.0, B), start=st, end=_en(en))
        return m

    ref = {ix: build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    rmet = {wn: ev(ref[ix], wn) for wn, ix in WINMAP}
    print("底座 G1B_B1：" + " ".join(
        "%s IC %.5f IR %.4f LS %.2f 压力 %.4f 换手 %.4f" % (wn, rmet[wn]["ic_mean"], rmet[wn]["ic_ir"],
                                                          rmet[wn]["ls_sharpe"], rmet[wn]["stress"], rmet[wn]["turnover"])
        for wn, _ in WINMAP))
    lams = a.lams.split(",") if a.lams else list(j["runs"])
    for lam in lams:
        run = j["runs"][lam]
        d = {k: float(v) for k, v in run["delta"].items() if abs(float(v)) > 1e-13}
        B = {ix: build(G, ix, "s161", d) for ix in (0, 1, 2)}
        cells, ok_all, det = [], True, []
        for wn, ix in WINMAP:
            m = ev(B[ix], wn)
            ok = all((m[k] - rmet[wn][k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in MET)
            ok_all &= ok
            cells.append("%+.4f%s" % (m["ic_mean"] - rmet[wn]["ic_mean"], "★" if ok else " "))
            det.append("%s IC %+.5f IR %+.4f LS %+.2f 压力 %+.4f 换手 %+.4f" %
                       (wn, m["ic_mean"] - rmet[wn]["ic_mean"], m["ic_ir"] - rmet[wn]["ic_ir"],
                        m["ls_sharpe"] - rmet[wn]["ls_sharpe"], m["stress"] - rmet[wn]["stress"],
                        m["turnover"] - rmet[wn]["turnover"]))
        print("\nλ=%s δ=%s" % (lam, json.dumps({k: round(v, 4) for k, v in d.items()})))
        print("   ΔIC " + "".join("%-9s" % x for x in cells) + ("  全★" if ok_all else ""))
        print("   " + " | ".join(det))
        if a.submit:
            name = "v_H3L%s" % lam.replace(".", "")
            G.write_submit(name, B, True)


if __name__ == "__main__":
    main()
