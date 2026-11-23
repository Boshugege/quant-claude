# -*- coding: utf-8 -*-
"""H1 族成员与现有头/根（1232、s161、1363）的逐截面秩相关（2023，判断是否为"新信息源"）。"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")


def main():
    c = R.core()
    m = ((pd.to_datetime(c.key_date).year == 2023) & np.isfinite(c.y)
         & (c.slot_of_date[c.date_id] != 930))
    idx = np.flatnonzero(m)
    did = c.date_id[idx]
    refs = {"1232": np.asarray(R.load("fm_1232")[0], float),
            "s161": np.asarray(O.axis(c, "fm_s161", "2020-01-01", "2024-12-31"), float),
            "1363": R.load("m_1363")[0] if False else None}
    for nm, arr in refs.items():
        if arr is None:
            continue
        arr = np.asarray(arr, float)
        fin = np.isfinite(arr[idx])
        print("%-6s 有效 %d" % (nm, fin.sum()))
    for f in sorted(OUT.glob("h1_*.npy")):
        x = np.load(f)[idx]
        line = "%-8s" % f.name[3:-4]
        for nm, arr in refs.items():
            if arr is None:
                continue
            ok = np.isfinite(arr[idx]) & np.isfinite(x)
            ii = idx[ok]
            d = pd.DataFrame({"did": c.date_id[ii], "a": x[ok], "b": arr[ii]})
            g = d.groupby("did")
            ra = g["a"].rank(pct=True)
            rb = g["b"].rank(pct=True)
            ra = ra - ra.groupby(d["did"]).transform("mean")
            rb = rb - rb.groupby(d["did"]).transform("mean")
            num = (ra * rb).groupby(d["did"]).sum()
            den = np.sqrt((ra * ra).groupby(d["did"]).sum() * (rb * rb).groupby(d["did"]).sum())
            cor = (num / den.replace(0, np.nan))
            n = g.size()
            cor = cor[(n >= 100) & cor.notna()]
            line += " |%s %+.3f" % (nm, cor.mean())
        print(line)


if __name__ == "__main__":
    main()
