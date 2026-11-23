# -*- coding: utf-8 -*-
"""精确快速评估器（第十三轮 G1/G3/G4/G5）。

平台口径：res = M · (B − mean)/sd(B)，M = I − U·Uᵀ（风格+行业）。只有 res 的**截面排名**进入指标，
而排名对截面内正常数缩放不变。于是对线性组合 B = Σ_i w_i x_i：

    M(B−mean(B))/sd(B)  ∝  Σ_i w_i · M(x_i − mean_i)  =  Σ_i w_i · sd_i · residual(x_i)

（mean/sd 都是该截面暴露行上的均值/总体标准差；对 x_i 先中心化再投影是线性的。）
所以只要预先对每个分量算一次 residual()，之后任意 w 的指标都可以用加权和 + 排名复现，**与
c.evaluate 完全一致**（已验证到 1e-12）。

用法：
  g_fast.py --plan p.json --ref 1363                # 打五窗口表
  g_fast.py --plan p.json --build NAME --win main --submit
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
WINS = [("2023", "2023-01-01", "2023-12-31"), ("2024", "2024-01-01", "2024-12-31"),
        ("2021", "2021-01-01", "2021-12-31"), ("2022", "2022-01-01", "2022-12-31"),
        ("2020", "2020-01-01", "2020-12-31")]
SUB = {"main": ("2023-01", "2024-12", ("2023-01-01", "2024-12-31"), "1015-203626-f5f0"),
       "o21": ("2021-01", "2022-12", ("2021-01-01", "2022-12-31"), "1017-102020-9a7c"),
       "o20": ("2020-01", "2020-12", ("2020-01-01", "2020-12-31"), "1017-102056-e592")}
ALL = ("2020-01-01", "2024-12-31")


def _en(s):
    return pd.Timestamp(s).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)


def inst_codes(c):
    m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", usecols=["local_id", "instrument"])
    num = m["local_id"].str.replace("ID", "", regex=False).astype(int)
    tab = pd.Series(m["instrument"].to_numpy(), index=num.to_numpy())
    return tab.reindex(c.inst).to_numpy()


class Fast:
    def __init__(self, plan):
        self.c = R.core()
        self.plan = plan
        self.dm = self.c.date_range_mask(*ALL)
        self.z930 = self.c.slot_of_date[self.c.date_id] == 930
        self.P = {}
        self.base_key = None

    def _stdg(self, x):
        c = self.c
        ii = c.has_idx
        g = c.date_id[ii]
        v = np.where(np.isfinite(x), x, 0.0)[ii]
        n = np.maximum(c.hcnt, 1).astype(float)
        s1 = np.bincount(g, v, c.n_dates)
        s2 = np.bincount(g, v * v, c.n_dates)
        var = np.maximum(s2 / n - (s1 / n) ** 2, 0.0)
        return np.sqrt(var)[c.date_id]      # 逐行

    def comp(self, key, arr):
        """arr 为原始值数组（已在 09:30 置 0）。返回 P = sd_i·residual(x_i)。

        分量在某截面内为常数时（sd_i=0）评测机的 residual 返回 NaN，但它在合成里贡献的就是 0，
        所以这里要把"截面内常数"补成 0（否则加权和会被 NaN 污染）。只有"该截面本身不参与评估"
        （暴露行 <50 或无基）与"该行没有官方暴露"才保留 NaN。"""
        if key not in self.P:
            c = self.c
            Z = c.residual(arr, self.dm)
            m = self._stdg(arr)
            ok_sec = ((c.hcnt >= 50) & (c.rank > 0))[c.date_id]
            keep = c.has & ok_sec
            self.P[key] = np.where(keep, np.where(np.isfinite(Z), m * np.nan_to_num(Z, nan=0.0), 0.0), np.nan)
        return self.P[key]

    def mix(self, w, mask=None):
        """w: dict key → 权重。mask = 底座原始值数组：合成文件的**行集就是底座的行集**，
        底座缺失的行在文件里不存在（平台按 0 处理），所以那里要把加权和压成 0。"""
        out = None
        for k, v in w.items():
            if abs(v) < 1e-13:
                continue
            out = v * self.P[k] if out is None else out + v * self.P[k]
        if mask is not None:
            out = np.where(np.isfinite(out), np.where(np.isfinite(mask), out, 0.0), np.nan)
        return out

    def win_of(self, s):
        day = s["day"]
        return {n: (day >= pd.Timestamp(a).value) & (day <= _en(b).value) for n, a, b in WINS}

    def score(self, arr, s=None):
        """返回 dict：整体(五窗口合并) 与逐窗口五项。"""
        if s is None:
            s = D.series_from_res(arr, *ALL)
        out = {}
        for n, _, _ in WINS:
            m = self.c.summarize(s, self.win_of(s)[n])
            out[n] = m
        return out


def load_plan(path, F):
    plan = json.load(open(path, encoding="utf-8"))
    heads = {}
    for hn, tm in plan["heads"].items():
        arrs = []
        for i in range(3):
            nm = tm[i] if len(tm) > i and tm[i] else ""
            arrs.append(R.load(nm)[0] if nm else np.full(F.c.n_rows, np.nan))
        for x in arrs:
            x[F.z930] = 0.0
        heads[hn] = arrs
        for i in range(3):
            F.comp("%s#%d" % (hn, i), arrs[i])
    F.heads = heads
    for r, al in plan["roots"].items():
        for i, pfx in enumerate(("fm_", "o21_", "o20_")):
            nm = r if r[:3] in ("fm_", "o2") else pfx + r
            try:
                ax = O.axis(F.c, nm, *(("2020-01-01", "2024-12-31") if i == 0 else ("2020-01-01", "2022-12-31")),
                            alpha=float(al or 0.0))
            except SystemExit:
                continue
            F.comp("%s#%d" % (r, i), ax)
    return plan, heads



def do_build(x, win, F, plan, heads, a, one):
    c = F.c
    rix = {"main": 0, "o21": 1, "o20": 2}[win]
    arr = F.mix(wvec_static(x["base"], x["w"], rix), heads[x["base"]][rix])
    st, en = SUB[win][2]
    din = (pd.to_datetime(c.key_date) >= pd.Timestamp(st)) & (pd.to_datetime(c.key_date) <= _en(en))
    o = pd.DataFrame({"date": pd.to_datetime(c.key_date), "instrument": inst_codes(c),
                      "factor": np.nan_to_num(arr, nan=0.0)})
    o = o[din & np.isfinite(arr)].drop_duplicates(["date", "instrument"], keep="last").reset_index(drop=True)
    f = OUT / (x["name"] + "_" + win + ".parquet")
    o.to_parquet(f, index=False)
    m = one(x["base"], x["w"], rix, {"main": "2023", "o21": "2021", "o20": "2020"}[win])
    print("%s [%s] IC %.5f IR %.4f LS %.2f 压力 %.4f 换手 %.4f n=%d rows=%d"
          % (x["name"], win, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], m["n_xsec"], len(o)))
    if a.submit:
        r0 = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values",
                             "-F", "start=%s" % SUB[win][0], "-F", "end=%s" % SUB[win][1],
                             "-F", "name=%s" % x["name"], "-F", "baseline=%s" % SUB[win][3], URL],
                            capture_output=True, text=True)
        print("  提交 %s/%s -> %s" % (x["name"], win, (r0.stdout.strip() or r0.stderr.strip())[:120]))
        if '"job_id"' in r0.stdout:
            f.unlink(missing_ok=True)


def wvec_static(base, w, ix):
    d = {base + "#" + str(ix): 1.0}
    for r, v in w.items():
        d[r + "#" + str(ix)] = d.get(r + "#" + str(ix), 0.0) + v
    return d



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--ref", default="")
    ap.add_argument("--build", default="")
    ap.add_argument("--build-all", action="store_true", dest="build_all")
    ap.add_argument("--wins", default="main,o21,o20")
    ap.add_argument("--win", default="main")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--name-out", default="")
    a = ap.parse_args()

    F = Fast(None)
    plan, heads = load_plan(a.plan, F)
    print("分量就绪：%d 个" % len(F.P))

    def wvec(base, w, ix):
        d = {base + "#" + str(ix): 1.0}
        for r, v in w.items():
            k = r + "#" + str(ix)
            if k in F.P:
                d[k] = d.get(k, 0.0) + v
        return d

    cache = {}

    def metrics(base, w, ix):
        key = (base, json.dumps(w, sort_keys=True), ix)
        if key not in cache:
            cache[key] = D.series_from_res(F.mix(wvec(base, w, ix), F.heads[base][ix]), *ALL)
        return cache[key]

    def one(base, w, ix, wn):
        s = metrics(base, w, ix)
        return F.c.summarize(s, F.win_of(s)[wn])

    combos = [x for x in plan["combos"] if a.only in x["name"]]
    if a.build_all:
        for x in combos:
            for win in [w for w in a.wins.split(",") if w]:
                do_build(x, win, F, plan, heads, a, one)
        return
    if a.build:
        x = [q for q in plan["combos"] if q["name"] == a.build][0]
        do_build(x, a.win, F, plan, heads, a, one)
        return

    MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")
    TOL = {"ic_mean": 2e-4, "ic_ir": 5e-3, "ls_sharpe": 0.3, "stress": 5e-3, "turnover": 0.0}
    print("\n%-18s %-5s| %-9s %-9s %-9s %-9s %-9s| %s" %
          ("combo", "win", "dIC", "dIR", "dLS", "dStress", "dTurn",
           "abs IC/IR/Turn   ★=对本底座五项不变差" + ("   （差 = 候选−%s）" % a.ref if a.ref else "")))
    for x in combos:
        for wn, _, _ in WINS:
            ix = 0 if wn in ("2023", "2024") else (1 if wn in ("2021", "2022") else 2)
            m = one(x["base"], x["w"], ix, wn)
            b = one(x["base"], {}, ix, wn)
            flag = "★" if all((m[k] - b[k]) * (1 if k != "turnover" else -1) >= -TOL[k] for k in MET) else " "
            if a.ref:
                rr = one(a.ref, {}, ix, wn)
                d = ["%+.5f" % (m[k] - rr[k]) for k in MET]
            else:
                d = ["%+.5f" % (m[k] - b[k]) for k in MET]
            print("%-18s %-5s| %s %s %s %s %s | %.5f/%.4f/%.4f   %s"
                  % (x["name"], wn, d[0], d[1], d[2], d[3], d[4], m["ic_mean"], m["ic_ir"], m["turnover"], flag))
        print()


if __name__ == "__main__":
    main()
