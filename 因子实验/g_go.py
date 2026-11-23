# -*- coding: utf-8 -*-
"""第十三轮 G2/G3/G4/G5 的统一评估器——**只用 c.evaluate（平台口径真值）**。

注意：g_fast 的 residual-分解口径在“非暴露行”上会给 0 而不是 NaN，导致 series_from_res 多算
485 个截面（o21 底座 IC 0.0811 vs 真值 0.0927）。本脚本一律走 原始合成值 + engine.Core.evaluate。

模式：
  g2  前一交易日（10 风格+申万一级行业）中性化，力度 λ
  g3  分槽位权重（s155 在每个截面一个权重）
  g4  换手前沿：1363 + δ·分量（含 s161 头部）
  g5  真实换手率根（g5_turn.py 产出的 npy）叠 1363
"""
import argparse, json, pathlib, subprocess, sys
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R
import d2_optimize as O
from g_opt import EXTRA

URL = "http://127.0.0.1:46220/api/jobs"
OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
HEADS = {"1363": ["1015-203626-f5f0", "1017-102020-9a7c", "1017-102056-e592"],
         "s161": ["fm_s161", "o21_s161", "o20_s161"],
         "s155": ["fm_s155", "o21_s155", "o20_s155"]}
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
IXW = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
TOL = {"ic_mean": 2e-4, "ic_ir": 5e-3, "ls_sharpe": 0.3, "stress": 5e-3, "turnover": 0.0}
SUB = {"main": ("2023-01", "2024-12", 0, "1015-203626-f5f0"),
       "o21": ("2021-01", "2022-12", 1, "1017-102020-9a7c"),
       "o20": ("2020-01", "2020-12", 2, "1017-102056-e592")}
EXP = "/home/user/jyc_scorer/data/exposure/exposure_%d.parquet"
STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY",
          "EARNYILD", "GROWTH", "LEVERAGE"]
SPEC = {l: (n1, n21, n20, al) for l, n1, n21, n20, al in (list(O.CAND) + list(EXTRA))}


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    return pd.Series(m["instrument"].to_numpy(), index=num.to_numpy()).reindex(c.inst).to_numpy()


class GO:
    def __init__(self):
        self.c = R.core()
        self.z930 = self.c.slot_of_date[self.c.date_id] == 930
        self.B = {}
        for hn, tm in HEADS.items():
            for ix in range(3):
                x = np.asarray(R.load(tm[ix])[0], float).copy()
                x[self.z930] = 0.0
                self.B[(hn, ix)] = x
        self.AX = {}

    def axis(self, label, ix):
        k = (label, ix)
        if k not in self.AX:
            n1, n21, n20, al = SPEC[label]
            nm = (n1, n21, n20)[ix]
            rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
            self.AX[k] = np.nan_to_num(O.axis(self.c, nm, *rng, alpha=float(al or 0.0)), nan=0.0)
        return self.AX[k]

    def evspan(self, B, st, en):
        m, _ = self.c.evaluate(np.where(self.z930, 0.0, B), start=st, end=_en(en))
        return m

    def ev(self, B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = self.c.evaluate(np.where(self.z930, 0.0, B), start=st, end=_en(en))
        return m

    def table(self, rows, ref="1363"):
        if not hasattr(self, "_refc"):
            self._refc = {}
        for wn, st, en in WINS:
            if wn not in self._refc:
                self._refc[wn] = self.evspan(self.B[(ref, IXW[wn])], st, en)
        print("%-22s|%s" % ("cand", "".join(" %-6s" % w for w, _, _ in WINS)))
        for nm, B in rows:
            cells = []
            for wn, _, _ in WINS:
                m = self.ev(B, wn)
                r = self._refc[wn]
                ok = all((m[k] - r[k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in MET)
                cells.append("%+.4f%s" % (m["ic_mean"] - r["ic_mean"], "★" if ok else " "))
            print("%-22s|%s" % (nm, "".join(" %-6s" % x for x in cells)))
        # 详细
        for nm, B in rows:
            for wn, _, _ in WINS:
                m = self.ev(B, wn)
                r = self.ev(self.B[(ref, IXW[wn])], wn)
                print("  %-18s %s IC %.5f IR %.4f LS %.2f 压力 %.4f 换手 %.4f | ΔIC %+.5f ΔIR %+.4f Δ压力 %+.4f Δ换手 %+.4f"
                      % (nm, wn, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"],
                         m["ic_mean"] - r["ic_mean"], m["ic_ir"] - r["ic_ir"], m["stress"] - r["stress"],
                         m["turnover"] - r["turnover"]))

    def write_submit(self, name, Bs, submit, mask_ix=0, mask_base="1363"):
        for win, B in Bs.items():
            s7, e7, ix, base_id = SUB[win]
            B = np.where(self.z930, 0.0, np.asarray(B, float))
            st, en = {"main": ("2023-01-01", "2024-12-31"), "o21": ("2021-01-01", "2022-12-31"),
                      "o20": ("2020-01-01", "2020-12-31")}[win]
            din = (pd.to_datetime(self.c.key_date) >= pd.Timestamp(st)) & (pd.to_datetime(self.c.key_date) <= _en(en))
            base = self.B[(mask_base, mask_ix if mask_ix < 0 else ix)]
            o = pd.DataFrame({"date": pd.to_datetime(self.c.key_date), "instrument": inst_codes(self.c),
                              "factor": np.nan_to_num(B, nan=0.0)})
            o = o[din & np.isfinite(base)].drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
            f = OUT / ("%s_%s.parquet" % (name, win))
            o.to_parquet(f, index=False)
            m = self.ev(B, {"main": "2023", "o21": "2021", "o20": "2020"}[win])
            print("%s [%s] IC %.5f IR %.4f 压力 %.4f 换手 %.4f rows=%d" %
                  (name, win, m["ic_mean"], m["ic_ir"], m["stress"], m["turnover"], len(o)))
            if submit:
                r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                                    "-F", "start=%s" % s7, "-F", "end=%s" % e7, "-F", "name=%s" % name,
                                    "-F", "baseline=%s" % base_id, URL], capture_output=True, text=True)
                print("  提交 %s/%s -> %s" % (name, win, (r.stdout.strip() or r.stderr.strip())[:140]))
                if '"job_id"' in r.stdout:
                    f.unlink(missing_ok=True)


def load_prev_expo(c):
    e = pd.concat([pd.read_parquet(EXP % y) for y in (2020, 2021, 2022, 2023, 2024)], ignore_index=True)
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e = e.drop_duplicates(["day", "instrument"], keep="last")
    ind = pd.get_dummies(e["industry_level1_code"], prefix="i").astype(np.float32)
    cols = STYLES + list(ind.columns)
    e = pd.concat([e[["day", "instrument"] + STYLES].reset_index(drop=True), ind.reset_index(drop=True)], axis=1)
    e = e.sort_values(["instrument", "day"])
    e["day"] = e.groupby("instrument")["day"].shift(-1)
    e = e.dropna(subset=["day"])[["day", "instrument"] + cols]
    need = pd.DataFrame({"row": np.arange(c.n_rows, dtype=np.int64),
                         "day": pd.to_datetime(c.day_of_date[c.date_id]), "instrument": inst_codes(c)})
    m = need.merge(e, on=["day", "instrument"], how="left", sort=False)
    V = np.full((c.n_rows, len(cols)), np.nan, np.float32)
    V[m["row"].to_numpy()] = m[cols].to_numpy(np.float32)
    return V, np.isfinite(V).all(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--name", default="")
    ap.add_argument("--penalty", default="3.0")
    ap.add_argument("--lams", default="0.5,1.0,1.5,2.0")
    ap.add_argument("--combo", default="G1B_B1")
    ap.add_argument("--ws", default="-0.05,-0.1,-0.2,-0.35")
    ap.add_argument("--roots", default="TNOW,TPREV")
    ap.add_argument("--base-w", type=float, default=0.25)
    ap.add_argument("--max-eval", type=int, default=56)
    ap.add_argument("--build-name", default="")
    a = ap.parse_args()
    G = GO()
    print("分量就绪")

    if a.mode == "g4":
        cfg = json.load(open(OUT / "g4_opt.json", encoding="utf-8"))["runs"]
        key = "%.4f" % float(a.penalty)
        delta = cfg[key]["delta"] if key in cfg else cfg[[k for k in cfg if abs(float(k) - float(a.penalty)) < 1e-9][0]]["delta"]
        rows = []
        Bs = {}
        for win, ix in (("main", 0), ("o21", 1), ("o20", 2)):
            B = np.asarray(G.B[("1363", ix)], float).copy()
            for lab, v in delta.items():
                if abs(v) < 1e-13:
                    continue
                if lab in ("s161", "s155", "s139"):
                    rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
                    B = B + v * np.nan_to_num(O.axis(G.c, HEADS[lab][ix], *rng), nan=0.0)
                else:
                    B = B + v * G.axis(lab, ix)
            Bs[win] = np.where(np.isfinite(G.B[("1363", ix)]), B, np.nan)
            rows.append(("G4_lam%s_%s" % (a.penalty, win), Bs[win]))
        G.table(rows)
        print("\n== G4 λ=%s 三区间 ==" % a.penalty)
        G.write_submit(a.build_name or ("v_G4L%s" % a.penalty.replace(".", "")), Bs, a.submit)
        return

    if a.mode == "g5":
        WS = [float(x) for x in a.ws.split(",")]
        for r in a.roots.split(","):
            arr = np.asarray(np.load(OUT / ("g5_%s.npy" % r)), float)
            arr = np.nan_to_num(O.rc_of(arr, G.c.date_id, G.c.n_dates), nan=0.0)   # 截面中心分位秩（与 O.axis 同口径）
            arr[G.z930] = 0.0
            rows = []
            for w in WS:
                Bs = {}
                for win, ix in (("main", 0), ("o21", 1), ("o20", 2)):
                    B = np.asarray(G.B[("1363", ix)], float).copy()
                    B = np.where(np.isfinite(B), B + w * np.nan_to_num(arr, nan=0.0), np.nan)
                    Bs[win] = B
                rows.append(("%s_w%+.2f" % (r, w), Bs))
                if a.build_name:
                    G.write_submit("%s_%s_w%+.2f" % (a.build_name, r, w), Bs, a.submit)
            if not a.build_name:
                for nm, Bs in rows:
                    print("== %s ==" % nm)
                    G.table([(nm + "_" + k, v) for k, v in Bs.items()])
        return

    if a.mode == "g2":
        c = G.c
        V, avail = load_prev_expo(c)
        print("前一交易日暴露可用 %d / %d" % (avail.sum(), c.n_rows))
        plan = json.load(open("/home/user/quant-claude/因子实验/plans/g2.json", encoding="utf-8"))
        x = [q for q in plan["combos"] if q["name"] == a.combo][0]
        basen = x["base"]
        print("底座 = %s" % basen)
        lams = [float(v) for v in a.lams.split(",")]
        dm = c.date_range_mask("2020-01-01", "2024-12-31")
        idx = np.flatnonzero(avail & dm[c.date_id] & ~G.z930)
        gg = c.date_id[idx]
        o = np.argsort(gg, kind="stable")
        idx, gg = idx[o], gg[o]
        st_ = np.searchsorted(gg, np.arange(c.n_dates)); en_ = np.searchsorted(gg, np.arange(c.n_dates), side="right")
        rows, Bs = [], {}
        for win, ix in (("main", 0), ("o21", 1), ("o20", 2)):
            B = np.asarray(G.B[(basen, ix)], float).copy()
            for r, w in x["w"].items():
                if abs(w) < 1e-13:
                    continue
                nm = r if r[:3] in ("fm_", "o2") else ("fm_", "o21_", "o20_")[ix] + r
                al = float(plan["roots"].get(r, 0.0) or 0.0)
                rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
                B = B + w * np.nan_to_num(O.axis(c, nm, *rng, alpha=al), nan=0.0)
            B = np.where(np.isfinite(G.B[(basen, ix)]), B, np.nan)
            rows.append(("G2_raw_" + win, B))
            out = {lam: np.full(c.n_rows, np.nan) for lam in lams}
            base = np.where(np.isfinite(B), B, 0.0)
            for d in range(c.n_dates):
                s, e = st_[d], en_[d]
                if e - s < 50:
                    continue
                ii = idx[s:e]
                xx = base[ii]
                sd = xx.std()
                if not sd > 0:
                    continue
                z = (xx - xx.mean()) / sd
                X = V[ii]
                b, *_ = np.linalg.lstsq(X, z, rcond=None)
                fit = X @ b
                for lam in lams:
                    out[lam][ii] = z - lam * fit
            for lam in lams:
                rows.append(("G2_lam%.2f_%s" % (lam, win), out[lam]))
            Bs[win] = out
        G.table(rows)
        if a.build_name:
            for lam in lams:
                print("== 提交 λ=%.2f ==" % lam)
                G.write_submit("%s_l%.2f" % (a.build_name, lam),
                               {k: v[lam] for k, v in Bs.items()}, a.submit, mask_base=basen)
        return

    if a.mode == "g3":
        c = G.c
        SL = [1000, 1030, 1100, 1130, 1330, 1400, 1430]
        axs = {}
        for ix in (0, 1, 2):
            rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
            ax = O.axis(c, ("fm_", "o21_", "o20_")[ix] + "s155", *rng)
            for s in SL:
                z = ax.copy(); z[c.slot_of_date[c.date_id] != s] = 0.0
                axs[(s, ix)] = np.nan_to_num(z, nan=0.0)
        print("槽位轴就绪")

        def build(w, ix):
            B = np.asarray(G.B[("1363", ix)], float).copy()
            for s in SL:
                if abs(w.get(s, 0.0)) > 1e-13:
                    B = B + w.get(s, 0.0) * axs[(s, ix)]
            return np.where(np.isfinite(G.B[("1363", ix)]), B, np.nan)

        cache = {}

        def fitJ(w):
            k = json.dumps({str(s): round(v, 5) for s, v in w.items()}, sort_keys=True)
            if k in cache:
                return cache[k]
            B = build(w, 1)
            Jm = G.evspan(B, "2021-01-01", "2022-12-31")
            Jm = {kk: Jm[kk] for kk in ("ic_mean", "ic_ir", "stress", "turnover", "n_xsec")}
            J = Jm["ic_ir"] + Jm["stress"] - 0.2 * Jm["turnover"]
            cache[k] = (J, Jm)
            return cache[k]

        w = {s: a.base_w for s in SL}
        J0, m0 = fitJ(w)
        print("固定 %.2f：fit J %.4f IC %.5f IR %.4f 压力 %.4f 换手 %.4f" %
              (a.base_w, J0, m0["ic_mean"], m0["ic_ir"], m0["stress"], m0["turnover"]))
        # 先拟合单一固定权重（分槽位的对照）
        bestf = (J0, a.base_w)
        for wv in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.5, 0.7):
            J, mm = fitJ({s: wv for s in SL})
            if J > bestf[0]:
                bestf = (J, wv)
        print("最优固定权重 %.3f：fit J %.4f" % (bestf[1], bestf[0]))
        best = (dict(w), J0, m0); nv = 0
        for it in range(3):
            for s in SL:
                cur = w[s]
                for sc in (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
                    if nv >= a.max_eval:
                        break
                    cand = dict(w); cand[s] = cur * sc
                    J, mm = fitJ(cand); nv += 1
                    if J > best[1]:
                        best = (cand, J, mm); w = cand
            print("  第 %d 轮：%s" % (it + 1, {s: round(v, 3) for s, v in w.items()}))
        print("评估 %d 次｜分槽位最优 %s" % (nv, json.dumps({str(s): round(v, 4) for s, v in best[0].items()})))
        print("  fit J %.4f IC %.5f IR %.4f 压力 %.4f 换手 %.4f" % (best[1], best[2]["ic_mean"], best[2]["ic_ir"], best[2]["stress"], best[2]["turnover"]))
        rows = []
        for wn, ix in (("2023", 0), ("2024", 0), ("2020", 2)):
            rows.append(("G3_slot_%s" % wn, build(best[0], ix)))
            rows.append(("G3_fixed_%s" % wn, build({s: bestf[1] for s in SL}, ix)))
            rows.append(("G3_base_%s" % wn, build({}, ix)))
        G.table(rows)
        if a.build_name:
            Bs = {win: build(best[0], ix) for win, ix in (("main", 0), ("o21", 1), ("o20", 2))}
            G.write_submit(a.build_name, Bs, a.submit)
        return


if __name__ == "__main__":
    main()
