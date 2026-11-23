# -*- coding: utf-8 -*-
"""通用 values 合成器（第十三轮 G1–G5）：B = 底座原始值 + Σ_j w_j · rank_j(±EWMA)。

- 底座可以是头部任务名（`--base s161` 自动按窗口选 fm_/o21_/o20_ 前缀），也可以是任务号 + `--base-o21/--base-o20`。
- 根用 `--roots "1232:+0.20@0.75,1236:+0.10,1237:-0.10"` 这种写法：`根名:权重[@EWMA α]`。
- rank 轴用 d2_optimize.axis（逐截面平均秩百分位−0.5，09:30 截面置 0），与已有的 values 合成脚本一致。
- 输出三列 parquet（date/instrument/factor），可 `--submit` 直接投评测机（kind=values）。
"""
import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402

URL = "http://127.0.0.1:46220/api/jobs"
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
WIN = {"main": ("2023-01-01", "2024-12-31", "2023-01", "2024-12"),
       "o21": ("2021-01-01", "2022-12-31", "2021-01", "2022-12"),
       "o20": ("2020-01-01", "2020-12-31", "2020-01", "2020-12")}
BASELINE = {"main": "1015-203626-f5f0", "o21": "1017-102020-9a7c", "o20": "1017-102056-e592"}


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv",
                    usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    tab = pd.Series(m["instrument"].to_numpy(), index=num.to_numpy())
    return tab.reindex(c.inst).to_numpy()


def parse_roots(s):
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        al = 0.0
        if "@" in tok:
            tok, _, a = tok.partition("@")
            al = float(a)
        nm, _, w = tok.partition(":")
        out.append((nm.strip(), float(w), al))
    return out


def task_name(root, win):
    return root if root[:3] in ("fm_", "o2") else {"main": "fm_" + root, "o21": "o21_" + root, "o20": "o20_" + root}[win]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-o21", default="")
    ap.add_argument("--base-o20", default="")
    ap.add_argument("--roots", default="")
    ap.add_argument("--delta-json", default="", help="g_opt 的 --out JSON（labels+delta），与 --roots 二选一")
    ap.add_argument("--win", default="main", choices=list(WIN))
    ap.add_argument("--name", required=True)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--baseline", default="")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    c = R.core()
    st, en, s7, e7 = WIN[a.win]
    roots = parse_roots(a.roots)
    if a.delta_json:
        j = json.load(open(a.delta_json, encoding="utf-8"))
        spec = {l: (n1, n21, n20, al) for l, n1, n21, n20, al in O.CAND}
        for lab, d in j["delta"].items():
            if abs(d) < 1e-9:
                continue
            n1, n21, n20, al = spec[lab]
            roots.append(({"main": n1, "o21": n21, "o20": n20}[a.win], d, al))
    if a.win == "main":
        bname = a.base if (a.base.startswith("fm_") or "-" in a.base) else "fm_" + a.base
    elif a.win == "o21":
        bname, bo21, bo20 = (a.base_o21 or ("o21_" + a.base)), "", ""
    else:
        bname, bo21, bo20 = (a.base_o20 or ("o20_" + a.base)), "", ""
    hf = R.load(bname)[0]
    B = np.asarray(hf, float).copy()
    for nm, w, al in roots:
        tn = task_name(nm, a.win)
        try:
            ax = O.axis(c, tn, st, en, al)
        except SystemExit:
            tn = task_name(nm, "main" if a.win == "o20" else "o21")
            ax = O.axis(c, tn, st, en, al)
            print("  !! %s 在 %s 窗口缺失，退回 %s" % (nm, a.win, tn))
        B = B + w * np.nan_to_num(ax, nan=0.0)
    B[c.slot_of_date[c.date_id] == 930] = 0.0
    m, _s = c.evaluate(B, start=st, end=en)
    print("%s [%s] base=%s IC %.5f IR %.4f LS %.2f 压力 %.4f 换手 %.4f (n=%d)"
          % (a.name, a.win, bname, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], m["n_xsec"]))
    if a.quiet:
        return
    o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                      "factor": np.nan_to_num(B, nan=0.0)})
    o = o[np.isfinite(np.asarray(hf, float))]
    o = o.drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / (a.name + "_" + a.win + ".parquet")
    o.to_parquet(f, index=False)
    print("写出 %s rows=%d" % (f, len(o)))
    if a.submit:
        base = a.baseline or BASELINE[a.win]
        r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                            "-F", "start=%s" % s7, "-F", "end=%s" % e7,
                            "-F", "name=%s" % a.name, "-F", "baseline=%s" % base, URL],
                           capture_output=True, text=True)
        print("提交 %s -> %s" % (a.name, r.stdout.strip() or r.stderr.strip()))
        if '"job_id"' in r.stdout:
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
