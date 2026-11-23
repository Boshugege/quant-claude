# -*- coding: utf-8 -*-
"""参照池：有公榜分数的因子的逐截面序列（存 store/refs/）。

2026-10-11 起参照池由**平台代码**构建（build_from_jobs）：把 因子代码/ 里实际提交到平台的 .py 在评分机上
按平台方式运行（kind=platform），再用这些任务的因子值算逐截面序列。旧版用 factor_lib 的 30 分钟面板复刻
（build_all，保留作对照），与平台实际跑的代码差异较大（例如 104 号两者截面相关只有 0.83），头部校准失效。

用途：
- 候选在参照池中的百分位、本地综合分；
- 用**同一时间快照**的公榜分校准“本地综合分 → 公榜分”（混用不同时间的快照会引入 0.02–0.05 的漂移）；
- 平台代码因子的原始值（float32），用于截面相关和配对比较的基线（ref:<编号>）。
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402

METRICS = ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]


def _chunk_job(args):
    a, b, fids = args
    import panel_store
    import factor_lib
    from factor_lib import add_components
    panel_store.configure_localeval()
    t0 = time.time()
    w = max(pd.Period(C.FULL_START, "M"), pd.Period(a, "M") - C.WARMUP_MONTHS)
    p = add_components(panel_store.load_panel(str(w), b))
    keep = p["is_eval"].to_numpy() & (p["day"] >= pd.Period(a, "M").start_time).to_numpy() & \
        (p["day"] <= pd.Period(b, "M").end_time).to_numpy()
    kd = p.loc[keep, "key_date"].to_numpy("datetime64[ns]").astype(np.int64)
    inst = p.loc[keep, "instrument_id"].to_numpy().astype(np.int32)
    fac = {str(x[0]): x for x in factor_lib.FACTORS}
    vals, errs = {}, {}
    for fid in fids:
        try:
            vals[fid] = np.asarray(fac[fid][3](p), np.float64)[keep].astype(np.float32)
        except Exception as e:                    # 个别因子在新 pandas 下可能报错，记录后跳过
            errs[fid] = f"{type(e).__name__}: {e}"
    out = C.REFS_DIR / "chunks" / f"{a}_{b}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, key_date=kd, inst=inst, **{f"f_{k}": v for k, v in vals.items()})
    return a, b, round(time.time() - t0), errs


def build_all(known, force=False, log=print, workers=2):
    import engine
    import panel_store
    C.REFS_DIR.mkdir(parents=True, exist_ok=True)
    fids = [str(x[0]) for x in known]
    todo = [(a, b, fids) for a, b in panel_store.year_chunks(C.FULL_START, C.FULL_END)
            if force or not (C.REFS_DIR / "chunks" / f"{a}_{b}.npz").exists()]
    errs = {}
    if todo:
        log(f"参照因子计算：{len(todo)} 个年度分块 × {len(fids)} 个因子，{workers} 进程")
        with ProcessPoolExecutor(workers) as ex:
            for a, b, sec, e in ex.map(_chunk_job, todo):
                errs.update(e)
                log(f"  {a}~{b} 完成，{sec}s；报错 {len(e)} 个")
    core = engine.Core(C.CORE_DIR, workers=C.ENGINE_THREADS)
    parts = [np.load(f) for f in sorted((C.REFS_DIR / "chunks").glob("*.npz"))]
    kd = np.concatenate([z["key_date"] for z in parts])
    inst = np.concatenate([z["inst"] for z in parts])
    info = {}
    allv = np.full((len(fids), core.n_rows), np.nan, np.float32)
    for i, (fid, name, score, _, note) in enumerate(known):
        fid = str(fid)
        if any(f"f_{fid}" not in z.files for z in parts):
            info[fid] = {"name": name, "public_score": score, "error": errs.get(fid, "部分年度缺失")}
            continue
        v = np.concatenate([z[f"f_{fid}"] for z in parts]).astype(np.float64)
        f, hit = core.align(kd, inst, v)
        allv[i] = f.astype(np.float32)
        s = core.series(f)
        np.savez(C.REFS_DIR / f"series_{fid}.npz", **{k: v for k, v in s.items() if not k.startswith("_")})
        m = core.summarize(s)
        info[fid] = {"name": name, "public_score": score, "note": note, "full_metrics": {k: m[k] for k in METRICS}}
        log(f"  {fid} {name}: IC {m['ic_mean']:.4f} IC_IR {m['ic_ir']:.3f} 压力 {m['stress']:.3f} 换手 {m['turnover']:.3f}")
    np.save(C.REFS_DIR / "values.npy", allv)
    (C.REFS_DIR / "refs.json").write_text(json.dumps({"fids": fids, "info": info}, ensure_ascii=False, indent=1), encoding="utf-8")


def build_from_jobs(manifest_path, log=print, jobs_dir=None):
    """由平台代码任务构建参照池。manifest：[{fid, name, public_score, snapshot, calib, job}]，job 为已完成的评分任务号。
    calib=True 的条目必须是同一时刻抄下的公榜分（同一快照），它们是公榜估计的锚点。
    现有参照池里任务号相同的因子直接复用序列和因子值，只更新公榜分、快照和 calib。
    先写到 store/refs_new/，完成后与 store/refs/ 互换（旧的改名为 refs_prev/）。"""
    import shutil
    import engine
    jobs_dir = Path(jobs_dir or C.JOBS)
    man = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = C.REFS_DIR.parent / "refs_new"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    core = engine.Core(C.CORE_DIR, workers=C.ENGINE_THREADS)
    old = {}                                         # 现有参照池里同一任务的因子直接复用（只更新公榜分/快照），不重算
    if (C.REFS_DIR / "refs.json").exists():
        om = json.loads((C.REFS_DIR / "refs.json").read_text(encoding="utf-8"))
        ov = np.load(C.REFS_DIR / "values.npy", mmap_mode="r")
        old = {f: (om["fids"].index(f), i) for f, i in om["info"].items() if f in om["fids"] and i.get("job")}
    fids, info, rows = [], {}, []
    for e in man:
        fid, jd = str(e["fid"]), jobs_dir / e["job"]
        if fid in old and old[fid][1]["job"] == e["job"] and (C.REFS_DIR / f"series_{fid}.npz").exists():
            i, oi = old[fid]
            shutil.copy2(C.REFS_DIR / f"series_{fid}.npz", out / f"series_{fid}.npz")
            fids.append(fid)
            rows.append(np.asarray(ov[i], np.float32))
            info[fid] = {**oi, "name": e["name"], "public_score": e["public_score"], "snapshot": e.get("snapshot"),
                         "calib": bool(e.get("calib"))}
            log(f"  {fid} {e['name']}: 复用（公榜 {e['public_score']}，快照 {e.get('snapshot')}）")
            continue
        kp = jd / "factor_keys.parquet"
        if not kp.exists():
            log(f"  {fid}: 任务 {e['job']} 没有因子值，跳过")
            continue
        k = pd.read_parquet(kp)
        f, hit = core.align(k["key_date"].to_numpy("datetime64[ns]").astype(np.int64), k["instrument_id"].to_numpy(), k["factor"].to_numpy())
        s = core.series(f)
        np.savez(out / f"series_{fid}.npz", **{kk: v for kk, v in s.items() if not kk.startswith("_")})
        m = core.summarize(s)
        res = json.loads((jd / "result.json").read_text(encoding="utf-8")) if (jd / "result.json").exists() else {}
        fids.append(fid)
        rows.append(f.astype(np.float32))
        info[fid] = {"name": e["name"], "public_score": e["public_score"], "snapshot": e.get("snapshot"), "calib": bool(e.get("calib")),
                     "source": "platform", "job": e["job"], "range": [res.get("meta", {}).get("start"), res.get("meta", {}).get("end")],
                     "warnings": res.get("warnings", []), "full_metrics": {kk: m[kk] for kk in METRICS}}
        log(f"  {fid} {e['name']}: IC {m['ic_mean']:.4f} IC_IR {m['ic_ir']:.3f} 压力 {m['stress']:.3f} 换手 {m['turnover']:.3f}")
    np.save(out / "values.npy", np.vstack(rows) if rows else np.zeros((0, core.n_rows), np.float32))
    (out / "refs.json").write_text(json.dumps({"fids": fids, "info": info, "source": "platform",
                                               "built": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=1), encoding="utf-8")
    prev = C.REFS_DIR.parent / "refs_prev"
    shutil.rmtree(prev, ignore_errors=True)
    if C.REFS_DIR.exists():
        C.REFS_DIR.rename(prev)
    out.rename(C.REFS_DIR)
    log(f"参照池已替换：{len(fids)} 个平台代码因子；旧参照池在 {prev}")


class RefPool:
    def __init__(self):
        meta = json.loads((C.REFS_DIR / "refs.json").read_text(encoding="utf-8"))
        self.fids, self.info = meta["fids"], meta["info"]
        self.source = meta.get("source", "panel")
        self.series = {}
        for fid in self.fids:
            p = C.REFS_DIR / f"series_{fid}.npz"
            if p.exists():
                z = np.load(p)
                self.series[fid] = {k: z[k] for k in z.files}
        self.values = np.load(C.REFS_DIR / "values.npy", mmap_mode="r")

    def values_for(self, fid):
        """参照因子在评分核心行上的原始值（float64），没有则返回 None。"""
        fid = str(fid)
        if fid not in self.fids or fid not in self.series:
            return None
        return np.asarray(self.values[self.fids.index(fid)], np.float64)

    def metrics(self, start, end):
        """各参照因子在 [start, end] 的五项指标。"""
        import engine
        lo, hi = pd.Timestamp(start).value, pd.Timestamp(end).value
        rows = {}
        for fid, s in self.series.items():
            m = (s["date"] >= lo) & (s["date"] <= hi)
            r = engine.Core.summarize(s, m)
            rows[fid] = {k: r[k] for k in METRICS}
        return pd.DataFrame(rows).T

    def anchors(self, start, end):
        """校准锚点：calib=True（同一公榜快照）的参照因子在 [start, end] 的五项指标 + 公榜分。"""
        t = self.metrics(start, end)
        t = t[[bool(self.info[f].get("calib")) and self.info[f].get("public_score") is not None for f in t.index]].dropna(subset=METRICS)
        t["public_score"] = [float(self.info[f]["public_score"]) for f in t.index]
        return t

    @staticmethod
    def _estimate(X, A, pub, k):
        """锚定近邻估计：对每个候选，取标准化指标空间里最近的 k 个锚点，
        估计 = 平均(锚点公榜分 + Σ PUB_BETA × (候选指标 − 锚点指标))。X: (n, 5)，A: (m, 5)。"""
        beta = np.array([C.PUB_BETA[c] for c in METRICS])
        sd = A.std(axis=0)
        sd[sd == 0] = 1.0
        out, near, d1 = np.full(len(X), np.nan), [], np.full(len(X), np.nan)
        for i, x in enumerate(X):
            if not np.all(np.isfinite(x)):
                near.append([])
                continue
            dist = np.sqrt((((x - A) / sd) ** 2).sum(axis=1))
            j = np.argsort(dist)[:k]
            out[i] = float(np.mean(pub[j] + (x - A[j]) @ beta))
            near.append(j.tolist())
            d1[i] = float(dist[j[0]])
        return out, near, d1

    def pub_estimate(self, cands: pd.DataFrame, start, end):
        """给一批候选（列含五项指标，索引任意；索引为参照池编号时视作留一，不用自己当锚点）估计公榜分。
        返回 (DataFrame[pub_est, pub_anchors], 校准信息)；锚点不足时返回 (None, None)。"""
        a = self.anchors(start, end)
        if len(a) < C.PUB_KNN + 2:
            return None, None
        cal = self.calibration(start, end, a)
        A, pub, names = a[METRICS].to_numpy(float), a["public_score"].to_numpy(float), list(a.index)
        est, anc, dist = np.full(len(cands), np.nan), [], np.full(len(cands), np.nan)
        for i, (idx, r) in enumerate(cands.iterrows()):
            keep = np.array([n != str(idx) for n in names])
            e, nn, d1 = self._estimate(r[METRICS].to_numpy(float)[None, :], A[keep], pub[keep], C.PUB_KNN)
            est[i], dist[i] = e[0], d1[0]
            kn = [n for n, m in zip(names, keep) if m]
            anc.append([kn[j] for j in nn[0]])
        return pd.DataFrame({"pub_est": est, "pub_anchors": anc, "pub_dist": dist,
                             "pub_extrap": dist > cal["extrap_dist"]}, index=cands.index), cal

    def calibration(self, start, end, a=None):
        """锚点留一法检验：每个锚点用其余锚点估计，报告平均误差和秩相关（全部 / 头部 ≥ PUB_HEAD）。"""
        key = (start, end)
        if getattr(self, "_cal_key", None) == key:
            return self._cal
        a = self.anchors(start, end) if a is None else a
        A, pub = a[METRICS].to_numpy(float), a["public_score"].to_numpy(float)
        loo = np.array([self._estimate(A[i:i + 1], np.delete(A, i, 0), np.delete(pub, i), C.PUB_KNN)[0][0] for i in range(len(a))])
        sd = A.std(axis=0)
        sd[sd == 0] = 1.0
        nn = np.sqrt((((A[:, None] - A[None]) / sd) ** 2).sum(axis=2))
        np.fill_diagonal(nn, np.inf)
        hd = pub >= C.PUB_HEAD
        rc = lambda x, y: float(pd.Series(x).rank().corr(pd.Series(y).rank())) if len(x) > 3 else float("nan")
        self._cal_key, self._cal = key, {
            "method": f"锚定近邻（k={C.PUB_KNN}，权重 PUB_BETA）", "snapshot": sorted({str(self.info[f].get("snapshot")) for f in a.index}),
            "n": int(len(a)), "loo_mae": float(np.mean(np.abs(loo - pub))), "loo_rank_corr": rc(loo, pub),
            "n_head": int(hd.sum()), "loo_mae_head": float(np.mean(np.abs(loo - pub)[hd])) if hd.any() else float("nan"),
            "loo_rank_corr_head": rc(loo[hd], pub[hd]), "beta": dict(C.PUB_BETA),
            "extrap_dist": float(np.quantile(nn.min(axis=1), 0.9)),   # 离最近锚点比 90% 的锚点彼此还远：外推，线性边际价值不再可靠
            "note": ("公榜分是全体提交中的百分位，样本外；本地换手、IC 均值几乎 1:1 搬到公榜，IC_IR/压力只能粗排，多空夏普不可信。"
                     "头部之间的先后本地基本分不出来（看 loo_rank_corr_head），差距小于 loo_mae_head 的两个候选应视为持平。"
                     "pub_extrap=True 表示指标落在锚点覆盖范围之外（例如换手远低于所有锚点），估计是线性外推，百分位会饱和，只能当方向参考。")}
        return self._cal

    def position(self, cand: dict, start, end):
        """候选因子在参照池中的百分位（换手越低越好）、本地综合分，以及锚定近邻估计的公榜分。"""
        t = self.metrics(start, end)
        t["public_score"] = [self.info[f]["public_score"] for f in t.index]
        t["name"] = [self.info[f]["name"] for f in t.index]
        t = t.dropna(subset=METRICS)
        if len(t) < 5:                               # 候选区间不在参照池覆盖范围内（参照池只跑了部分区间）
            return None
        allm = pd.concat([t[METRICS], pd.DataFrame([cand], index=["__cand__"])[METRICS]])
        pct = pd.DataFrame(index=allm.index)
        for c in METRICS:
            pct[c] = (-allm[c] if c == "turnover" else allm[c]).rank(pct=True)
        comp = pct.mean(axis=1)
        t["local_composite"] = comp.drop("__cand__")
        pe, calib = self.pub_estimate(pd.DataFrame([cand], index=["__cand__"]), start, end)
        est = float(pe.loc["__cand__", "pub_est"]) if pe is not None else float("nan")
        extrap = bool(pe.loc["__cand__", "pub_extrap"]) if pe is not None else None
        anchors = [{"fid": f, "name": self.info[f]["name"], "public_score": self.info[f]["public_score"]}
                   for f in (pe.loc["__cand__", "pub_anchors"] if pe is not None else [])]
        calib = calib or {}
        rho = calib.get("loo_rank_corr", float("nan"))
        beat = {c: float(((allm.loc["__cand__", c] < t[c]) if c == "turnover" else (allm.loc["__cand__", c] > t[c])).mean())
                for c in METRICS}
        top = t.sort_values("public_score", ascending=False).head(8)
        return {"n_refs": int(len(t)), "percentile": {c: float(pct.loc["__cand__", c]) for c in METRICS},
                "share_of_refs_beaten": beat, "local_composite": float(comp["__cand__"]),
                "calibration_rank_corr": rho, "estimated_public_score": est, "estimate_anchors": anchors, "estimate_extrapolated": extrap,
                "calibration": calib, "source": self.source,
                "note": ("百分位是在本地参照池（" + ("平台代码在评分机上运行的结果" if self.source == "platform" else "factor_lib 30 分钟面板复刻，头部不可信")
                         + "，同一区间同一口径）内的排名。估计公榜分 = 指标最接近的几个同快照锚点的公榜分 + 指标差 × 边际价值（换手、IC 均值为主），"
                         f"头部留一法平均误差 {calib.get('loo_mae_head', float('nan')):.3f}、头部秩相关 {calib.get('loo_rank_corr_head', float('nan')):.2f}："
                         "头部之间的先后本地基本分不出来。平台分数是全体提交中的排名，会随别人提交变化。"),
                "top_refs": [{"fid": f, "name": r["name"], "public_score": r["public_score"],
                              **{c: float(r[c]) for c in METRICS}, "local_composite": float(r["local_composite"])}
                             for f, r in top.iterrows()]}

    def correlations(self, core, f, start, end, fids=("90", "104", "107", "105")):
        dm = core.date_range_mask(start, end)
        out = {}
        for fid in fids:
            if fid in self.fids:
                i = self.fids.index(fid)
                out[fid] = {"name": self.info[fid]["name"],
                            "rank_corr_mean": core.per_date_rank_corr(f, np.asarray(self.values[i], np.float64), dm)}
        return out


if __name__ == "__main__":
    # python refs.py build <manifest.json>：按清单重建参照池（服务运行中可执行，完成后调 POST /api/admin/reload_refs）
    if len(sys.argv) == 3 and sys.argv[1] == "build":
        build_from_jobs(sys.argv[2])
    else:
        print("用法：python refs.py build <manifest.json>")
