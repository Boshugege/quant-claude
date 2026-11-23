# -*- coding: utf-8 -*-
"""G2：合成之后再中性化（用**前一交易日**的申万一级行业 + 10 个风格暴露）。

对合成值 B 逐截面：z = (B − mean)/sd（暴露行），X = [1, 10 风格, 行业哑变量]（前一交易日），
残差 = z − λ·X·b（b = lstsq）。λ∈{0.5,1,1.5,2}。合成口径与 g_batch 一致（底座原始值 + Σ w·rank）。

  g_neut.py --plan 因子实验/plans/g1.json --combo G1B_B1 --lams 0.5,1,1.5,2
  g_neut.py --plan ... --combo G1B_B1 --lams 1.0 --win main --submit --name-out v_G2A
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
import d2_eval as D  # noqa: E402
import d2_optimize as O  # noqa: E402

URL = "http://127.0.0.1:46220/api/jobs"
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
EXP = "/home/user/jyc_scorer/data/exposure/exposure_%d.parquet"
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
          "EARNYILD", "GROWTH", "LEVERAGE"]
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
ALL = ("2020-01-01", "2024-12-31")
SUB = {"main": ("2023-01", "2024-12", "1015-203626-f5f0"), "o21": ("2021-01", "2022-12", "1017-102020-9a7c"),
       "o20": ("2020-01", "2020-12", "1017-102056-e592")}


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    tab = pd.Series(m["instrument"].to_numpy(), index=num.to_numpy())
    return tab.reindex(c.inst).to_numpy()


def load_prev_expo(c):
    """逐 core 行的前一交易日 [10 风格, 行业哑变量]（没有则 NaN）。用 merge，不用 reindex
    （core 每个 (day, inst) 有 8 行、176 个代码对应多个伪代码，reindex 会因重复索引报错）。"""
    e = pd.concat([pd.read_parquet(EXP % y) for y in (2020, 2021, 2022, 2023, 2024)], ignore_index=True)
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e = e.drop_duplicates(["day", "instrument"], keep="last")
    ind = pd.get_dummies(e["industry_level1_code"], prefix="i").astype(np.float32)
    cols = STYLES + list(ind.columns)
    e = pd.concat([e[["day", "instrument"] + STYLES].reset_index(drop=True),
                   ind.reset_index(drop=True)], axis=1)
    e = e.sort_values(["instrument", "day"])
    e["day"] = e.groupby("instrument")["day"].shift(-1)      # 该行适用的（下一）交易日
    e = e.dropna(subset=["day"])[["day", "instrument"] + cols]
    need = pd.DataFrame({"row": np.arange(c.n_rows),
                         "day": pd.to_datetime(c.day_of_date[c.date_id]),
                         "instrument": inst_codes(c)})
    m = need.merge(e, on=["day", "instrument"], how="left", sort=False)
    row = m["row"].to_numpy()
    V = np.full((c.n_rows, len(cols)), np.nan, np.float32)
    V[row] = m[cols].to_numpy(np.float32)
    return V, np.isfinite(V).all(axis=1)


def build_composite(plan, name, c, ix):
    x = [q for q in plan["combos"] if q["name"] == name]
    if not x:
        raise SystemExit("plan 里没有 %s" % name)
    x = x[0]
    tm = plan["heads"][x["base"]]
    nm = tm[ix] if len(tm) > ix and tm[ix] else ""
    B = np.asarray(R.load(nm)[0], float).copy()
    B[c.slot_of_date[c.date_id] == 930] = 0.0
    for r, w in x["w"].items():
        if abs(w) < 1e-13:
            continue
        al = float(plan["roots"].get(r, 0.0) or 0.0)
        tn = r if r[:3] in ("fm_", "o2") else {0: "fm_", 1: "o21_", 2: "o20_"}[ix] + r
        B = B + w * np.nan_to_num(O.axis(c, tn, *ALL, alpha=al), nan=0.0)
    return B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--combo", required=True)
    ap.add_argument("--lams", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--ix", type=int, default=0, help="0=主区间底座, 1=2021-22, 2=2020")
    ap.add_argument("--win", default="main")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--name-out", default="")
    ap.add_argument("--save", default="")
    a = ap.parse_args()

    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    plan = json.load(open(a.plan, encoding="utf-8"))
    B = build_composite(plan, a.combo, c, a.ix)
    V, avail = load_prev_expo(c)
    print("前一交易日暴露可用行 %d / %d" % (avail.sum(), c.n_rows))

    lams = [float(x) for x in a.lams.split(",")]
    dm = c.date_range_mask(*ALL)
    idx = np.flatnonzero(avail & dm[c.date_id] & ~z930)
    g = c.date_id[idx]
    order = np.argsort(g, kind="stable")
    idx, g = idx[order], g[order]
    starts = np.searchsorted(g, np.arange(c.n_dates))
    ends = np.searchsorted(g, np.arange(c.n_dates), side="right")
    base = np.where(np.isfinite(B), B, 0.0)
    out = {lam: np.full(c.n_rows, np.nan) for lam in lams}
    for d in range(c.n_dates):
        s, e = starts[d], ends[d]
        if e - s < 50:
            continue
        ii = idx[s:e]
        x = base[ii]
        sd = x.std()
        if not sd > 0:
            continue
        z = (x - x.mean()) / sd
        X = V[ii]
        b, *_ = np.linalg.lstsq(X, z, rcond=None)
        fit = X @ b
        for lam in lams:
            out[lam][ii] = z - lam * fit
    # 基准（λ=0）指标
    s0 = D.series_from_res(c.residual(B, dm), *ALL)
    r0 = {n: c.summarize(s0, (s0["day"] >= pd.Timestamp(st).value) & (s0["day"] <= _en(en).value)) for n, st, en in WINS}
    print("λ=0.00(原始)  " + " | ".join("%s IC %.5f IR %.4f 换手 %.4f 压力 %.4f"
                                     % (n, r0[n]["ic_mean"], r0[n]["ic_ir"], r0[n]["turnover"], r0[n]["stress"]) for n, _, _ in WINS))
    sers = {}
    for lam in lams:
        s = D.series_from_res(out[lam], *ALL)
        sers[lam] = s
        r = {n: c.summarize(s, (s["day"] >= pd.Timestamp(st).value) & (s["day"] <= _en(en).value)) for n, st, en in WINS}
        md = ["%s ΔIC %+.5f ΔIR %+.4f Δ换手 %+.4f Δ压力 %+.4f "
              % (n, r[n]["ic_mean"] - r0[n]["ic_mean"], r[n]["ic_ir"] - r0[n]["ic_ir"],
                 r[n]["turnover"] - r0[n]["turnover"], r[n]["stress"] - r0[n]["stress"]) for n, _, _ in WINS]
        allok = all(r[n][k] - r0[n][k] >= -{"ic_mean": 2e-4, "ic_ir": 5e-3, "ls_sharpe": 0.3, "stress": 5e-3, "turnover": 0.0}[k]
                    for n, _, _ in WINS for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"))
        print("λ=%.2f %s  %s" % (lam, "★" if allok else " ", " ".join(md)))
    if a.save:
        np.save(OUT / (a.save + ".npy"), out[lams[0]])
    if a.submit:
        lam = lams[0]
        arr = out[lam]
        c2 = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
              "o20": ("2020-01-01", "2020-12-31")}[a.win]
        din = (pd.to_datetime(c.key_date) >= pd.Timestamp(c2[0])) & (pd.to_datetime(c.key_date) <= _en(c2[1]))
        o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                          "factor": np.nan_to_num(arr, nan=0.0)})
        o = o[din & np.isfinite(arr)].drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
        OUT.mkdir(parents=True, exist_ok=True)
        f = OUT / ((a.name_out or "g2") + ".parquet")
        o.to_parquet(f, index=False)
        print("写出", f, len(o))
        r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                            "-F", "start=%s" % SUB[a.win][0], "-F", "end=%s" % SUB[a.win][1],
                            "-F", "name=%s" % (a.name_out or "g2"), "-F", "baseline=%s" % SUB[a.win][2], URL],
                           capture_output=True, text=True)
        print("提交 ->", r.stdout.strip() or r.stderr.strip())
        if '"job_id"' in r.stdout:
            f.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
