# -*- coding: utf-8 -*-
"""G5 备选二：按前一交易日风格分组的**条件信号**（用户提示的第 4 类新空间）。

变体：
  A. 在「前一交易日 SIZE 十分位」内对合成值去组均值（去掉规模的非线性部分）
  B. 在「前一交易日 SIZE 十分位」内把合成值换成组内中心分位秩
  C. 在「前一交易日申万一级行业」内去组均值
  D. 在「前一交易日 RESVOL 五分位」内去组均值
一律只用前一交易日的暴露（错位），评测机随后再做一次当天 10 风格 + 32 行业线性剔除。
"""
import argparse, pathlib, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R
import engine as E
import d2_optimize as O

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
EXP = "/home/user/jyc_scorer/data/exposure/exposure_%d.parquet"
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
          "EARNYILD", "GROWTH", "LEVERAGE"]
HEADS = {"1363": ["1015-203626-f5f0", "1017-102020-9a7c", "1017-102056-e592"]}
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
IXW = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    return pd.Series(m["instrument"].to_numpy(), index=num.to_numpy()).reindex(c.inst).to_numpy()


def load_expo(c):
    e = pd.concat([pd.read_parquet(EXP % y) for y in (2020, 2021, 2022, 2023, 2024)], ignore_index=True)
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e = e.drop_duplicates(["day", "instrument"], keep="last")
    ind = pd.get_dummies(e["industry_level1_code"], prefix="i").astype(np.float32)
    e = pd.concat([e[["day", "instrument"] + STYLES].reset_index(drop=True), ind.reset_index(drop=True)], axis=1)
    cols = STYLES + [q for q in e.columns if q.startswith("i_")]
    e = e.sort_values(["instrument", "day"])
    e["day"] = e.groupby("instrument")["day"].shift(-1)
    e = e.dropna(subset=["day"])[["day", "instrument"] + cols]
    need = pd.DataFrame({"row": np.arange(c.n_rows, dtype=np.int64),
                         "day": pd.to_datetime(c.day_of_date[c.date_id]), "instrument": inst_codes(c)})
    m = need.merge(e, on=["day", "instrument"], how="left", sort=False)
    V = np.full((c.n_rows, len(cols)), np.nan, np.float32)
    V[m["row"].to_numpy()] = m[cols].to_numpy(np.float32)
    return V, cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="A,B,C,D")
    a = ap.parse_args()
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    B = {}
    for ix in range(3):
        x = np.asarray(R.load(HEADS["1363"][ix])[0], float).copy()
        x[z930] = 0.0
        B[ix] = x
    V, cols = load_expo(c)
    av = np.isfinite(V).all(axis=1)
    print("暴露可用 %d / %d" % (av.sum(), c.n_rows))
    dm = c.date_range_mask("2020-01-01", "2024-12-31")
    ii = np.flatnonzero(av & dm[c.date_id] & ~z930)
    gid = c.date_id[ii]
    sz = V[ii, cols.index("SIZE")]
    rv = V[ii, cols.index("RESVOL")]
    ind_cols = [j for j, q in enumerate(cols) if q.startswith("i_")]
    indid = np.argmax(V[ii][:, ind_cols], axis=1)
    K = c.n_dates
    size_dec = np.minimum((E._group_rank_pct(gid, sz, K)[0] * 10).astype(int), 9)
    rv_dec = np.minimum((E._group_rank_pct(gid, rv, K)[0] * 5).astype(int), 4)
    keys = {"A": gid * 10 + size_dec, "B": gid * 10 + size_dec, "C": gid * 40 + indid, "D": gid * 5 + rv_dec}
    Ks = {k: int(v.max()) + 1 for k, v in keys.items()}

    def demean(Bx, key, Kx):
        w = np.isfinite(Bx[ii]).astype(float)
        s1 = np.bincount(key, np.nan_to_num(Bx[ii], nan=0.0) * w, Kx)
        s0 = np.bincount(key, w, Kx)
        out = Bx.copy()
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ii] = np.where(np.isfinite(Bx[ii]), Bx[ii] - (s1 / np.maximum(s0, 1))[key], np.nan)
        return out

    def rankin(Bx, key, Kx):
        out = np.full(c.n_rows, np.nan)
        sel = ii[np.isfinite(Bx[ii])]
        rq, _ = E._group_rank_pct(key[np.isfinite(Bx[ii])], Bx[sel], Kx)
        out[sel] = rq - 0.5
        return out

    ref = {}
    for wn, st, en in WINS:
        ref[wn] = c.evaluate(np.where(z930, 0.0, B[IXW[wn]]), start=st, end=_en(en))[0]
    print("\n%-6s %-5s| ΔIC ΔIR Δ压力 Δ换手  (★=该窗口五项不变差)")
    for v in a.variants.split(","):
        k, Kx = keys[v], Ks[v]
        ok = 0
        cells = []
        for wn, st, en in WINS:
            ix = IXW[wn]
            Bx = rankin(B[ix], k, Kx) if v == "B" else demean(B[ix], k, Kx)
            m = c.evaluate(np.where(z930, 0.0, Bx), start=st, end=_en(en))[0]
            r = ref[wn]
            good = (m["ic_mean"] - r["ic_mean"] >= -2e-4 and m["ic_ir"] - r["ic_ir"] >= -5e-3
                    and m["stress"] - r["stress"] >= -5e-3 and m["turnover"] - r["turnover"] <= 0)
            ok += good
            cells.append("%s %+.5f/%+.4f/%+.4f/%+.4f%s" % (wn, m["ic_mean"] - r["ic_mean"], m["ic_ir"] - r["ic_ir"],
                        m["stress"] - r["stress"], m["turnover"] - r["turnover"], "★" if good else " "))
        print("%-6s (%d/5) | %s" % (v, ok, " | ".join(cells)))
    print("DONE_G5C")


if __name__ == "__main__":
    main()
