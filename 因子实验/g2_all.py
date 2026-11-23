# -*- coding: utf-8 -*-
"""G2（修正版）：合成后再中性化。对每个区间用**对应底座**（main/o21/o20）各做一次
前一交易日（10 风格 + 申万一级行业）中性化，再拼出五窗口表。

残差 = z − λ·X·b，z 为合成值在暴露行上的截面 z 分数，X 为**前一交易日**的暴露（错位）。
判定窗口与底座一一对应：ix=0→2023/2024，ix=1→2021/2022，ix=2→2020。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_eval as D  # noqa: E402
import d2_optimize as O  # noqa: E402

EXP = "/home/user/jyc_scorer/data/exposure/exposure_%d.parquet"
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
          "EARNYILD", "GROWTH", "LEVERAGE"]
ALL = ("2020-01-01", "2024-12-31")
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
IXWIN = {0: ("2023", "2024"), 1: ("2021", "2022"), 2: ("2020",)}
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
TOL = {"ic_mean": 2e-4, "ic_ir": 5e-3, "ls_sharpe": 0.3, "stress": 5e-3, "turnover": 0.0}


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    tab = pd.Series(m["instrument"].to_numpy(), index=num.to_numpy())
    return tab.reindex(c.inst).to_numpy()


def load_prev_expo(c):
    e = pd.concat([pd.read_parquet(EXP % y) for y in (2020, 2021, 2022, 2023, 2024)], ignore_index=True)
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e = e.drop_duplicates(["day", "instrument"], keep="last")
    ind = pd.get_dummies(e["industry_level1_code"], prefix="i").astype(np.float32)
    cols = STYLES + list(ind.columns)
    e = pd.concat([e[["day", "instrument"] + STYLES].reset_index(drop=True),
                   ind.reset_index(drop=True)], axis=1)
    e = e.sort_values(["instrument", "day"])
    e["day"] = e.groupby("instrument")["day"].shift(-1)          # 该行适用的（下一）交易日
    e = e.dropna(subset=["day"])[["day", "instrument"] + cols]
    need = pd.DataFrame({"row": np.arange(c.n_rows, dtype=np.int64),
                         "day": pd.to_datetime(c.day_of_date[c.date_id]),
                         "instrument": inst_codes(c)})
    m = need.merge(e, on=["day", "instrument"], how="left", sort=False)
    V = np.full((c.n_rows, len(cols)), np.nan, np.float32)
    V[m["row"].to_numpy()] = m[cols].to_numpy(np.float32)
    return V, np.isfinite(V).all(axis=1)


def build_composite(plan, name, c, ix):
    x = [q for q in plan["combos"] if q["name"] == name][0]
    tm = plan["heads"][x["base"]]
    B = np.asarray(R.load(tm[ix])[0], float).copy()
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
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    plan = json.load(open(a.plan, encoding="utf-8"))
    V, avail = load_prev_expo(c)
    print("前一交易日暴露可用行 %d / %d" % (avail.sum(), c.n_rows))
    lams = [float(x) for x in a.lams.split(",")]
    dm = c.date_range_mask(*ALL)
    idx_all = np.flatnonzero(avail & dm[c.date_id] & ~z930)
    gall = c.date_id[idx_all]
    order = np.argsort(gall, kind="stable")
    idx_all, gall = idx_all[order], gall[order]
    starts = np.searchsorted(gall, np.arange(c.n_dates))
    ends = np.searchsorted(gall, np.arange(c.n_dates), side="right")

    tab = {}          # (combo) -> {lam -> {win -> metrics}}
    for ix in (0, 1, 2):
        B = build_composite(plan, a.combo, c, ix)
        base = np.where(np.isfinite(B), B, 0.0)
        out = {lam: np.full(c.n_rows, np.nan) for lam in lams}
        for d in range(c.n_dates):
            s, e = starts[d], ends[d]
            if e - s < 50:
                continue
            ii = idx_all[s:e]
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
        s0 = D.series_from_res(c.residual(B, dm), *ALL)
        r0 = {n: c.summarize(s0, (s0["day"] >= pd.Timestamp(st).value) & (s0["day"] <= _en(en).value))
              for n, st, en in WINS if n in IXWIN[ix]}
        tab[("base", ix)] = r0
        for lam in lams:
            s = D.series_from_res(out[lam], *ALL)
            tab[(lam, ix)] = {n: c.summarize(s, (s["day"] >= pd.Timestamp(st).value) & (s["day"] <= _en(en).value))
                              for n, st, en in WINS if n in IXWIN[ix]}
        print("ix=%d(%s) 原始 %s" % (ix, ",".join(IXWIN[ix]),
              " ".join("%s IC %.5f IR %.4f 换手 %.4f 压力 %.4f" % (n, r0[n]["ic_mean"], r0[n]["ic_ir"],
                                                                  r0[n]["turnover"], r0[n]["stress"]) for n in IXWIN[ix])))
        del out

    print("\n== %s 五窗口（相对未中性化）==" % a.combo)
    for lam in lams:
        parts, allok = [], True
        for n, _, _ in WINS:
            ix = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}[n]
            m, b = tab[(lam, ix)][n], tab[("base", ix)][n]
            ok = all((m[k] - b[k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in MET)
            allok &= ok
            parts.append("%s ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f%s"
                         % (n, m["ic_mean"] - b["ic_mean"], m["ic_ir"] - b["ic_ir"],
                            m["stress"] - b["stress"], m["turnover"] - b["turnover"], "★" if ok else " "))
        print("λ=%.2f %s %s" % (lam, "ALL★" if allok else "   ", " | ".join(parts)))
    if a.out:
        ser = {}
        for (k, ix), v in tab.items():
            key = "base" if k == "base" else "lam%.2f" % k
            for n, mm in v.items():
                ser["%s|%s|%s" % (a.combo, key, n)] = {kk: float(mm[kk]) for kk in
                                                       ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover", "n_xsec")}
        json.dump(ser, open(a.out, "w"), ensure_ascii=False, indent=1)
        print("写出", a.out)


if __name__ == "__main__":
    main()
