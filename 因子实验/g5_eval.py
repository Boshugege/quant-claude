# -*- coding: utf-8 -*-
"""G5 评估：把 g5_turn.py 产出的真实换手率根（npy）叠到底座上，五窗口判五项。

用 g_fast 的精确口径（逐分量 residual × 截面 sd，再求和）；判定与本项目一致：
IC ≥ −2e-4、IC_IR ≥ −5e-3、压力 ≥ −5e-3、换手 ≤ 0 才算该窗口 ★。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_eval as D  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
ALL = ("2020-01-01", "2024-12-31")
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
HEADS = {"1363": ["1015-203626-f5f0", "1017-102020-9a7c", "1017-102056-e592"]}


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def Pmix(c, x, dm):
    Z = c.residual(x, dm)
    ii = c.has_idx
    g = c.date_id[ii]
    v = np.where(np.isfinite(x), x, 0.0)[ii]
    n = np.maximum(c.hcnt, 1).astype(float)
    s1 = np.bincount(g, v, c.n_dates)
    s2 = np.bincount(g, v * v, c.n_dates)
    sd = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0.0))[c.date_id]
    ok_sec = ((c.hcnt >= 50) & (c.rank > 0))[c.date_id]
    keep = c.has & ok_sec
    return np.where(keep, np.where(np.isfinite(Z), sd * np.nan_to_num(Z, nan=0.0), 0.0), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="1363")
    ap.add_argument("--roots", default="TNOW,TPREV,TREL,AUCK")
    ap.add_argument("--ws", default="0.1,0.2,0.35")
    ap.add_argument("--mask", action="store_true", default=True)
    a = ap.parse_args()

    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    dm = c.date_range_mask(*ALL)
    b = {}
    for i, nm in enumerate(HEADS[a.base]):
        x = R.load(nm)[0].copy()
        x[z930] = 0.0
        b[i] = x
    Pb = {i: Pmix(c, b[i], dm) for i in b}
    sers = {}
    for i in b:
        sers[i] = D.series_from_res(Pb[i], *ALL)
    bm = {i: {n: c.summarize(sers[i], (sers[i]["day"] >= pd.Timestamp(st).value)
                              & (sers[i]["day"] <= _en(en).value)) for n, st, en in WINS} for i in b}
    WS = [float(x) for x in a.ws.split(",")]
    for r in a.roots.split(","):
        arr = np.load(OUT / ("g5_%s.npy" % r))
        arr[z930] = 0.0
        Pa = Pmix(c, arr, dm)
        sa = D.series_from_res(Pa, *ALL)
        row = []
        for n, st, en in WINS:
            m = c.summarize(sa, (sa["day"] >= pd.Timestamp(st).value) & (sa["day"] <= _en(en).value))
            row.append("%s IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (n, m["ic_mean"], m["ic_ir"], m["stress"], m["turnover"]))
        print("单独 %-6s %s" % (r, " | ".join(row)))
        cro = c.per_date_rank_corr(np.nan_to_num(Pa, nan=0.0), np.nan_to_num(Pb[0], nan=0.0),
                                   c.date_range_mask("2023-01-01", "2024-12-31"))
        print("        与 1363 的逐截面秩相关均值 %.3f（2023–24）" % cro)
        for w in WS:
            out = []
            for n, st, en in WINS:
                i = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}[n]
                mix = Pb[i] + w * Pa
                mix = np.where(np.isfinite(mix), np.where(np.isfinite(b[i]), mix, 0.0), np.nan)
                s = D.series_from_res(mix, *ALL)
                m = c.summarize(s, (s["day"] >= pd.Timestamp(st).value) & (s["day"] <= _en(en).value))
                mb = bm[i][n]
                ok = (m["ic_mean"] - mb["ic_mean"] >= -2e-4 and m["ic_ir"] - mb["ic_ir"] >= -5e-3
                      and m["stress"] - mb["stress"] >= -5e-3 and m["turnover"] - mb["turnover"] <= 0)
                out.append("%s ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f%s"
                           % (n, m["ic_mean"] - mb["ic_mean"], m["ic_ir"] - mb["ic_ir"],
                              m["stress"] - mb["stress"], m["turnover"] - mb["turnover"], "★" if ok else " "))
            print("  w=%.2f %s" % (w, " | ".join(out)))


if __name__ == "__main__":
    main()
