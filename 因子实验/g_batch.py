# -*- coding: utf-8 -*-
"""批量合成评估器（第十三轮 G1–G5 共用）。

一次进程里加载若干底座（三个区间的任务）与若干根的 rank 轴（d2_optimize.axis，09:30 置 0），
再对一批组合 B = 底座值 + Σ w_j·rank_j 在五个窗口（2023／2024／2021／2022／2020）上算五项，
打印相对参照底座（默认 1363）与相对自身底座的差（★ = 该窗口五项都不变差）。

plan.json：
{ "heads": {"1363": ["1015-203626-f5f0","1017-102020-9a7c","1017-102056-e592"],
            "s161": ["fm_s161","o21_s161","o20_s161"]},        # [主区间, 2021-22, 2020]
  "roots": {"1232": 0.75, "1236": 0.0},                        # 值 = EWMA α
  "ref": "1363",
  "combos": [{"name":"G1B","base":"s161","w":{"1232":0.2,...}}] }

区间→任务：2023/2024 用 heads[w][0] 与 fm_<root>；2021/2022 用 heads[w][1] 与 o21_<root>；
2020 用 heads[w][2] 与 o20_<root>。
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
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
SLOTIX = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
PREFIX = {0: "fm_", 1: "o21_", 2: "o20_"}
SUBMIT_WIN = {"main": ("2023-01", "2024-12", 0, "1015-203626-f5f0"),
              "o21": ("2021-01", "2022-12", 1, "1017-102020-9a7c"),
              "o20": ("2020-01", "2020-12", 2, "1017-102056-e592")}


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    tab = pd.Series(m["instrument"].to_numpy(), index=num.to_numpy())
    return tab.reindex(c.inst).to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--build", default="", help="把该组合写成 parquet")
    ap.add_argument("--win", default="main")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--name-out", default="")
    a = ap.parse_args()

    plan = json.load(open(a.plan, encoding="utf-8"))
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    HB, RA = {}, {}

    def base(w):
        if w not in HB:
            tm = plan["heads"][w]
            arrs = []
            for i in range(3):
                nm = tm[i] if len(tm) > i and tm[i] else ""
                arrs.append(R.load(nm)[0] if nm else np.full(c.n_rows, np.nan))
            for x in arrs:
                x[z930] = 0.0
            HB[w] = arrs
        return HB[w]

    def root(r, ix):
        key = (r, ix)
        if key not in RA:
            al = float(plan["roots"].get(r, 0.0) or 0.0)
            nm = r if r[:3] in ("fm_", "o2") else PREFIX[ix] + r
            RA[key] = O.axis(c, nm, "2020-01-01", "2022-12-31" if ix else "2024-12-31", alpha=al)
        return RA[key]

    def mix(bn, w, ix):
        B = np.asarray(base(bn)[ix], float).copy()
        for r, wv in w.items():
            if abs(wv) < 1e-12:
                continue
            B = B + wv * np.nan_to_num(root(r, ix), nan=0.0)
        return B

    def evwin(B, ix, st, en):
        m, _ = c.evaluate(np.where(z930, 0.0, B), start=st, end=_en(en))
        return m

    def submit(name, bn, w, win):
        ix = SLOTIX[[k for k in SLOTIX if k == win][0]] if win in SLOTIX else SUBMIT_WIN[win][2]
        s7, e7, ix, base_id = SUBMIT_WIN[win]
        B = mix(bn, w, ix)
        B[z930] = 0.0
        st, en = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
                  "o20": ("2020-01-01", "2020-12-31")}[win]
        din = (pd.to_datetime(c.key_date) >= pd.Timestamp(st)) & (pd.to_datetime(c.key_date) <= _en(en))
        o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                          "factor": np.nan_to_num(B, nan=0.0)})
        o = o[din & np.isfinite(np.asarray(base(bn)[ix], float))]
        o = o.drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / (name + ".parquet")
        o.to_parquet(f, index=False)
        st, en = [q for q in WINS if q[0] == ("2023" if win == "main" else win)][0][1:3]
        if win == "main":
            st, en = "2023-01-01", "2024-12-31"
        m = evwin(B, ix, st, _en(en))
        print("%s [%s] IC %.5f IR %.4f LS %.2f 压力 %.4f 换手 %.4f n=%d -> %s"
              % (name, win, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], m["n_xsec"], f))
        if a.submit:
            r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                                "-F", "start=%s" % s7, "-F", "end=%s" % e7,
                                "-F", "name=%s" % name, "-F", "baseline=%s" % base_id, URL],
                               capture_output=True, text=True)
            print("  提交 ->", r.stdout.strip() or r.stderr.strip())
            if '"job_id"' in r.stdout:
                f.unlink(missing_ok=True)

    hits = [x for x in plan["combos"] if a.only in x["name"]]
    if a.build:
        x = [q for q in plan["combos"] if q["name"] == a.build]
        if not x:
            raise SystemExit("plan 里没有 %s" % a.build)
        submit(a.name_out or a.build, x[0]["base"], x[0]["w"], a.win)
        return

    MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
    TOL = {"ic_mean": 2e-4, "ic_ir": 5e-3, "ls_sharpe": 0.3, "stress": 5e-3, "turnover": 0.0}
    cache = {}

    def metrics(bn, ix, wn):
        k = (bn, wn)
        if k not in cache:
            st, en = [q for q in WINS if q[0] == wn][0][1:3]
            cache[k] = evwin(np.asarray(base(bn)[ix], float), ix, st, en)
        return cache[k]

    ref = plan.get("ref", "")
    if ref:
        print("参照 = %s（下表前三列差为 候选−参照）\n" % ref)
    print("%-18s %-5s| %-9s %-9s %-9s %-9s %-9s| %s" %
          ("combo", "win", "dIC", "dIR", "dLS", "dStress", "dTurn", "abs IC/IR/Turn  ★=对本底座五项不变差"))
    for x in hits:
        for wn, st, en in WINS:
            ix = SLOTIX[wn]
            m = evwin(mix(x["base"], x["w"], ix), ix, st, en)
            b = metrics(x["base"], ix, wn)
            flag = "★" if all((m[k] - b[k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in MET) else " "
            if ref:
                rr = metrics(ref, ix, wn)
                d = ["%+.5f" % (m[k] - rr[k]) for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")]
            else:
                d = ["%+.5f" % (m[k] - b[k]) for k in MET]
            print("%-18s %-5s| %s %s %s %s %s | %.5f/%.4f/%.4f  %s"
                  % (x["name"], wn, d[0], d[1], d[2], d[3], d[4], m["ic_mean"], m["ic_ir"], m["turnover"], flag))
        print()


if __name__ == "__main__":
    main()
