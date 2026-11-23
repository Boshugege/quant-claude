# -*- coding: utf-8 -*-
"""H1 最好版本提交：1399（G4LOW）+ w·rank(L_5)，baseline 用 G4LOW 三区间。"""
import argparse
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
from g_go import GO, inst_codes, SUB  # noqa: E402
from h1_stack import build, g4low_delta  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
URL = "http://127.0.0.1:46220/api/jobs"
BASE = {"main": "1017-125252-c6a7", "o21": "1017-125258-063c", "o20": "1017-125308-0167"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--member", default="L_5")
    ap.add_argument("--w", type=float, default=0.15)
    ap.add_argument("--name", default="v_H1L5G4")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--base", default="G4LOW")
    a = ap.parse_args()
    G = GO()
    c = G.c
    sig = np.nan_to_num(O.rc_of(np.load(OUT / ("h1_%s.npy" % a.member)), c.date_id, c.n_dates), nan=0.0)
    d = g4low_delta()
    Bs = {}
    for win, ix in (("main", 0), ("o21", 1), ("o20", 2)):
        B = build(G, ix, "1363", d) + a.w * sig
        Bs[win] = np.where(G.z930, 0.0, B)
        m = G.ev(Bs[win], {"main": "2023", "o21": "2021", "o20": "2020"}[win])
        print("%s IC %.5f IR %.4f LS %.2f 压力 %.4f 换手 %.4f" %
              (win, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"]))
    for win, B in Bs.items():
        s7, e7, ix, _ = SUB[win]
        st, en = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
                  "o20": ("2020-01-01", "2020-12-31")}[win]
        en_ts = pd.Timestamp(en).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        din = (pd.to_datetime(c.key_date) >= pd.Timestamp(st)) & (pd.to_datetime(c.key_date) <= en_ts)
        base = G.B[("1363", ix)]
        o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                          "factor": np.nan_to_num(B, nan=0.0)})
        o = o[din & np.isfinite(base)].drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
        f = OUT / ("%s_%s.parquet" % (a.name, win))
        o.to_parquet(f, index=False)
        print("%s rows=%d" % (win, len(o)))
        if a.submit:
            r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                                "-F", "start=%s" % s7, "-F", "end=%s" % e7, "-F", "name=%s" % a.name,
                                "-F", "baseline=%s" % BASE[win], URL], capture_output=True, text=True)
            print("  提交 %s/%s -> %s" % (a.name, win, (r.stdout.strip() or r.stderr.strip())[:160]))
            if '"job_id"' in r.stdout:
                f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
