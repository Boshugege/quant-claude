# -*- coding: utf-8 -*-
"""H2 参照：平台已提交的线性解 v_D2OPT3_raw（三区间）与 v_G4LOW（三区间）五项。"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
from g_go import GO, WINS  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
G4 = {"main": "1017-125252-c6a7", "o21": "1017-125258-063c", "o20": "1017-125308-0167"}
WIN = [("2023", "main"), ("2024", "main"), ("2021", "o21"), ("2022", "o21"), ("2020", "o20")]


def load_dir(jid):
    d = pathlib.Path("/home/user/jyc_scorer/jobs") / jid
    kf = pd.read_parquet(d / "factor_keys.parquet")
    return R.core().align(kf["key_date"].to_numpy("datetime64[ns]").astype(np.int64),
                          kf["instrument_id"].to_numpy(), kf["factor"].to_numpy(float))[0]


def main():
    G = GO()
    c = G.c
    A = {"D2": {"main": R.load("v_D2OPT3_raw")[0], "o21": R.load("v_D2OPT3O21_raw")[0],
                "o20": R.load("v_D2OPT3O20_raw")[0]},
         "G4": {k: load_dir(v) for k, v in G4.items()}}
    for nm, arrs in A.items():
        line = "%-4s" % nm
        for wn, win in WIN:
            st, en = [q for q in WINS if q[0] == wn][0][1:3]
            en = pd.Timestamp(en).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
            m, _ = c.evaluate(np.asarray(arrs[win], float), start=st, end=en)
            line += " |%s %.5f/%.4f/%.4f/%.4f" % (wn, m["ic_mean"], m["ic_ir"], m["stress"], m["turnover"])
        print(line)


if __name__ == "__main__":
    main()
