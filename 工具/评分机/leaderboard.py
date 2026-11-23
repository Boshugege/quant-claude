# -*- coding: utf-8 -*-
"""赋分表：按平台的赋分规则，给我们自己的全部因子排名。

平台分数 = 五项指标（IC 均值、IC_IR、多空夏普、压力稳定性、换手率）各自在全体有效提交中的百分位，等权平均。
赋分表把“全体提交”换成“我们自己登记过的全部因子”：
- 参照池里有公榜分数的因子（平台代码在评分机上运行的结果）；
- 评分机上所有已完成、用本地官方标签（y_off）、区间覆盖赋分窗口、且通过平台规则检查的任务（平台会报错的提交不给分，也不参与排名）。

每个任务评分时保存逐截面序列 series.npz，赋分表从中切出统一窗口的五项指标，所以不同区间提交的任务也能同场比较。
同一个因子提交多次会出现多行；name 相同的任务只保留最新一次（dedupe=True）。

和平台的差别：总体只有我们自己的因子，104 号的变体很多，头部格外拥挤，
所以这里的分差会比公榜大；它用来衡量进步和找出各项短板，不是公榜分的估计。
公榜估计另列 pub_est：用参照池里同一快照的公榜分做锚点（refs.RefPool.pub_estimate），sort="pub_est" 按它排序。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import engine  # noqa: E402

METRICS = ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]
SERIES_KEYS = ("g", "date", "day", "slot", "ic", "ls", "ysd", "n", "tl", "ts")
DEFAULT_WINDOW = ("2023-01", "2024-12")


def save_series(job_dir: Path, s: dict):
    np.savez(job_dir / "series.npz", **{k: s[k] for k in SERIES_KEYS})


def _window_mask(s, start, end):
    lo, hi = pd.Period(start, "M").start_time.value, pd.Period(end, "M").end_time.value
    return (s["date"] >= lo) & (s["date"] <= hi)


def excluded():
    """store/excluded.json：{任务号: 原因}。列出的任务不参与赋分表（例如用了未来数据），结果仍保留在 jobs/。"""
    p = C.STORE / "excluded.json"
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        return {}


def entries(refpool, start, end, dedupe=True, include_refs=True):
    rows = []
    skip = excluded()
    if include_refs and refpool is not None:
        for fid, s in refpool.series.items():
            m = _window_mask(s, start, end)
            if m.sum() < 100:
                continue
            r = engine.Core.summarize(s, m)
            info = refpool.info[fid]
            rows.append({"key": f"ref:{fid}", "name": f"{fid} {info['name']}", "source": "参照池",
                         "public_score": info.get("public_score"), "snapshot": info.get("snapshot"),
                         "created": None, "n_xsec": r["n_xsec"], **{k: r[k] for k in METRICS}})
    jobs = []
    ref_jobs = {i.get("job") for i in refpool.info.values()} if refpool is not None else set()   # 已作为参照池行出现的任务
    for d in C.JOBS.iterdir():
        sp, st = d / "series.npz", d / "state.json"
        if d.name in skip or d.name in ref_jobs or not (sp.exists() and st.exists()):
            continue
        try:
            state = json.loads(st.read_text(encoding="utf-8"))
            spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        if state.get("status") != "done" or spec.get("label", "y_off") != "y_off":
            continue
        if not (spec["start"] <= start and spec["end"] >= end):          # 任务区间必须覆盖赋分窗口
            continue
        if (spec.get("name") or "").startswith("ref_"):                  # 参照池构建任务已经以“参照池”行出现
            continue
        fail = state.get("platform_fail")
        if fail is None and (d / "result.json").exists():                # 旧任务没有这个标记，从结果里读平台规则检查
            pc = json.loads((d / "result.json").read_text(encoding="utf-8")).get("platform_checks") or {}
            fail = bool(pc) and not pc.get("pass", True)
        if fail or state.get("data_violation"):                           # 平台会报错、不给分，或读了禁用表的提交不参与排名
            continue
        jobs.append((state, spec, sp))
    if dedupe:
        latest = {}
        for state, spec, sp in jobs:
            k = state.get("name") or state["id"]
            if k not in latest or state["created"] > latest[k][0]["created"]:
                latest[k] = (state, spec, sp)
        jobs = list(latest.values())
    for state, spec, sp in jobs:
        z = np.load(sp)
        s = {k: z[k] for k in z.files}
        m = _window_mask(s, start, end)
        if m.sum() < 100:
            continue
        r = engine.Core.summarize(s, m)
        rows.append({"key": state["id"], "name": state.get("name") or state["id"], "source": spec["kind"],
                     "public_score": None, "snapshot": None, "created": state.get("created"), "n_xsec": r["n_xsec"],
                     **{k: r[k] for k in METRICS}})
    return rows


def table(refpool, start=DEFAULT_WINDOW[0], end=DEFAULT_WINDOW[1], dedupe=True, include_refs=True, sort="score"):
    rows = entries(refpool, start, end, dedupe, include_refs)
    if not rows:
        return {"window": [start, end], "n": 0, "rows": []}
    t = pd.DataFrame(rows).dropna(subset=METRICS).reset_index(drop=True)
    for k in METRICS:                                                     # 与平台相同：百分位排名，换手越低越好
        t[f"pct_{k}"] = (-t[k] if k == "turnover" else t[k]).rank(pct=True, method="average")
    t["score"] = t[[f"pct_{k}" for k in METRICS]].mean(axis=1)
    t = t.sort_values("score", ascending=False).reset_index(drop=True)
    t["rank"] = np.arange(1, len(t) + 1)
    X = np.column_stack([t[k].to_numpy() * (-1 if k == "turnover" else 1) for k in METRICS])
    t["pareto"] = [not np.any(np.all(X >= X[i], axis=1) & np.any(X > X[i], axis=1)) for i in range(len(t))]
    calib = None
    if refpool is not None:                                               # 公榜估计：锚定近邻（参照池行用留一，不拿自己当锚点）
        idx = [k[4:] if str(k).startswith("ref:") else k for k in t["key"]]
        pe, calib = refpool.pub_estimate(pd.DataFrame(t[METRICS].to_numpy(), index=idx, columns=METRICS), start, end)
        if pe is not None:
            t["pub_est"] = pe["pub_est"].to_numpy()
            t["pub_anchors"] = list(pe["pub_anchors"])
            t["pub_extrap"] = pe["pub_extrap"].to_numpy()
            t["pub_rank"] = t["pub_est"].rank(ascending=False, method="min")
    if sort == "pub_est" and "pub_est" in t:                              # rank 始终是赋分名次，pub_rank 是公榜估计名次
        t = t.sort_values("pub_est", ascending=False, na_position="last").reset_index(drop=True)
    leaders = {k: t.loc[t[k].idxmin() if k == "turnover" else t[k].idxmax(), "name"] for k in METRICS}
    weakest = {}
    for i, r in t.iterrows():                                            # 每个因子五项里百分位最低的一项（最该补的短板）
        p = {k: r[f"pct_{k}"] for k in METRICS}
        weakest[r["key"]] = min(p, key=p.get)
    t["weakest"] = t["key"].map(weakest)
    out = t.replace({np.nan: None}).to_dict("records")
    return {"window": [start, end], "n": len(t), "leaders": leaders, "rows": out, "sort": sort, "pub_calibration": calib,
            "note": "赋分 = 五项指标在本表全部因子中的百分位等权平均（换手越低越好），与平台规则相同，但总体只有我们自己的因子；"
                    "用来衡量进步和找短板，不是公榜分的估计；公榜估计看 pub_est（误差见 pub_calibration）。pareto = 没有其他因子在五项上同时不差且至少一项更好。"}


def backfill(log=print):
    """给已完成但缺 series.npz 的任务补算逐截面序列（用任务保存的因子值，不重跑因子代码）。"""
    core = engine.Core(C.CORE_DIR, workers=C.ENGINE_THREADS)
    n = 0
    for d in sorted(C.JOBS.iterdir()):
        if (d / "series.npz").exists() or not (d / "factor_keys.parquet").exists() or not (d / "result.json").exists():
            continue
        try:
            spec = json.loads((d / "spec.json").read_text(encoding="utf-8"))
            if spec.get("label", "y_off") != "y_off":
                continue
            k = pd.read_parquet(d / "factor_keys.parquet")
            f, _ = core.align(k["key_date"].to_numpy("datetime64[ns]").astype(np.int64), k["instrument_id"].to_numpy(), k["factor"].to_numpy())
            dm = core.date_range_mask(pd.Period(spec["start"], "M").start_time, pd.Period(spec["end"], "M").end_time)
            save_series(d, core.series(f, True, dm))
            n += 1
        except Exception as e:
            log(f"  {d.name}: {type(e).__name__}: {e}")
    log(f"补算 {n} 个任务的逐截面序列")


if __name__ == "__main__":
    if sys.argv[1:] == ["backfill"]:
        backfill()
