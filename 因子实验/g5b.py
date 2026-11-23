# -*- coding: utf-8 -*-
"""G5 备选：用暴露表构造「前一交易日风格暴露差分」根（ΔSTYLE），叠到 1363 上判五窗口。

口径（严格错位）：截面日 t 只用 t−1 与 t−2 两条暴露行 → Δ = STYLE(t−1) − STYLE(t−2)。
ΔSIZE ≈ 个股相对收益（log 流通市值的变化）→ −ΔSIZE 就是「昨日收益反转」；
ΔLIQUIDTY / ΔRESVOL / ΔBTOP 同理是全新的日频信息源。
"""
import argparse, pathlib, subprocess, sys
import numpy as np, pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R
import d2_optimize as O

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
URL = "http://127.0.0.1:46220/api/jobs"
EXP = "/home/user/jyc_scorer/data/exposure/exposure_%d.parquet"
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
          "EARNYILD", "GROWTH", "LEVERAGE"]
HEADS = {"1363": ["1015-203626-f5f0", "1017-102020-9a7c", "1017-102056-e592"]}
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
IXW = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
SUB = {"main": ("2023-01", "2024-12", 0, "1015-203626-f5f0"),
       "o21": ("2021-01", "2022-12", 1, "1017-102020-9a7c"),
       "o20": ("2020-01", "2020-12", 2, "1017-102056-e592")}


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    return pd.Series(m["instrument"].to_numpy(), index=num.to_numpy()).reindex(c.inst).to_numpy()


def load_dstyle(c):
    e = pd.concat([pd.read_parquet(EXP % y) for y in (2020, 2021, 2022, 2023, 2024)], ignore_index=True)
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e = e.drop_duplicates(["day", "instrument"], keep="last").sort_values(["instrument", "day"])
    d = e[["day", "instrument"]].copy()
    for s in STYLES:
        d["d" + s] = e.groupby("instrument")[s].diff()               # Δ 在原生日上算
    cols = ["d" + s for s in STYLES]
    d = d.dropna(subset=["dSIZE"])                                   # 首日无 Δ
    d["day"] = d.groupby("instrument")["day"].shift(-1)              # 适用日 = 下一交易日
    d = d.dropna(subset=["day"])
    need = pd.DataFrame({"row": np.arange(c.n_rows, dtype=np.int64),
                         "day": pd.to_datetime(c.day_of_day if False else c.day_of_date[c.date_id]),
                         "instrument": inst_codes(c)})
    m = need.merge(d[["day", "instrument"] + cols], on=["day", "instrument"], how="left", sort=False)
    X = np.full((c.n_rows, len(cols)), np.nan, np.float32)
    X[m["row"].to_numpy()] = m[cols].to_numpy(np.float32)
    return {cols[j][1:]: X[:, j] for j in range(len(cols))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default="SIZE,LIQUIDTY,RESVOL,BTOP")
    ap.add_argument("--ws", default="-0.1,-0.2,0.1,0.2")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--build-name", default="")
    ap.add_argument("--build-field", default="")
    ap.add_argument("--build-w", type=float, default=0.0)
    a = ap.parse_args()
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    B = {}
    for ix in range(3):
        x = np.asarray(R.load(HEADS["1363"][ix])[0], float).copy()
        x[z930] = 0.0
        B[ix] = x
    DS = load_dstyle(c)
    print("Δ 暴露可用率：" + " ".join("%s %.1f%%" % (k, 100 * np.isfinite(v).mean()) for k, v in DS.items()))
    AX = {}
    for k, v in DS.items():
        ax = np.nan_to_num(O.rc_of(v, c.date_id, c.n_dates), nan=0.0)
        ax[z930] = 0.0
        AX[k] = ax

    ref = {}
    for wn, st, en in WINS:
        ref[wn] = c.evaluate(np.where(z930, 0.0, B[IXW[wn]]), start=st, end=_en(en))[0]

    def ev(Bx, wn, st, en):
        return c.evaluate(np.where(z930, 0.0, Bx), start=st, end=_en(en))[0]

    WS = [float(x) for x in a.ws.split(",")]
    print("\n%-16s %-5s| %s" % ("cand", "win", "IC / IR / LS / 压力 / 换手   ΔIC ΔIR Δ压力 Δ换手"))
    best = []
    for f in a.fields.split(","):
        for w in WS:
            ok = 0
            line = []
            for wn, st, en in WINS:
                ix = IXW[wn]
                Bx = np.where(np.isfinite(B[ix]), B[ix] + w * AX[f], np.nan)
                m = ev(Bx, wn, st, en)
                r = ref[wn]
                good = (m["ic_mean"] - r["ic_mean"] >= -2e-4 and m["ic_ir"] - r["ic_ir"] >= -5e-3
                        and m["stress"] - r["stress"] >= -5e-3 and m["turnover"] - r["turnover"] <= 0)
                ok += good
                line.append("%s %+.5f/%+.4f/%+.4f/%+.4f%s" % (wn, m["ic_mean"] - r["ic_mean"],
                            m["ic_ir"] - r["ic_ir"], m["stress"] - r["stress"],
                            m["turnover"] - r["turnover"], "★" if good else " "))
            print("%-16s w=%+.2f | %s" % (f, w, " | ".join(line)))
            best.append((ok, f, w))
    best.sort(reverse=True)
    print("\n最好：%s" % best[:5])
    if a.build_name:
        f, w = a.build_field, a.build_w
        for win, ix in (("main", 0), ("o21", 1), ("o20", 2)):
            s7, e7, _, base_id = SUB[win]
            Bx = np.where(np.isfinite(B[ix]), B[ix] + w * AX[f], np.nan)
            Bx = np.where(z930, 0.0, Bx)
            st, en = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
                      "o20": ("2020-01-01", "2020-12-31")}[win]
            din = (pd.to_datetime(c.key_date) >= pd.Timestamp(st)) & (pd.to_datetime(c.key_date) <= _en(en))
            o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                              "factor": np.nan_to_num(Bx, nan=0.0)})
            o = o[din & np.isfinite(B[ix])].drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
            p = OUT / ("%s_%s.parquet" % (a.build_name, win))
            o.to_parquet(p, index=False)
            print("%s [%s] rows=%d" % (a.build_name, win, len(o)))
            if a.submit:
                rr = subprocess.run(["curl", "-s", "-F", "files=@%s" % p, "-F", "kind=values", "-F", "start=%s" % s7,
                                     "-F", "end=%s" % e7, "-F", "name=%s" % a.build_name, "-F", "baseline=%s" % base_id,
                                     URL], capture_output=True, text=True)
                print("  提交 ->", (rr.stdout.strip() or rr.stderr.strip())[:140])
                if '"job_id"' in rr.stdout:
                    p.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
