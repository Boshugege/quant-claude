# -*- coding: utf-8 -*-
"""本地因子研究工具：时间切分 + 固定参照池评分，用于控制泛化性。

规则（详见 README.md“防过拟合”一节）：
- 开发集 2020-01~2022-12：探索与调参；
- 验证集 2023-01~2023-12：挑选候选；
- 保留集 2024-01~2024-12：只在最终确认时使用一次；
- 候选需在开发集与验证集上同时优于基准才入选；
- 评分时把每个候选单独放入“固定参照池”（factor_lib.FACTORS 中的全部因子）计算百分位，
  候选之间互不影响，结果可复现。
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import localeval as le  # noqa: E402
from factor_lib import FACTORS, add_components  # noqa: E402

SPLITS = {
    "开发": ("2020-01", "2022-12"),
    "验证": ("2023-01", "2023-12"),
    "保留": ("2024-01", "2024-12"),
}
METRICS = ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "结果", "缓存")

_PANELS = {}


def panel_for(split):
    """读取并缓存某个切分的面板（含全部中间量）。"""
    if split not in _PANELS:
        start, end = SPLITS[split]
        p = add_components(le.build_panel(le.load_bars(start, end)))
        _PANELS[split] = p
    return _PANELS[split]


def metrics_of(series, split, label="y_off"):
    m = le.evaluate(series, panel_for(split), label=label)
    return {k: m[k] for k in METRICS}


def pool_metrics(split, label="y_off", n_jobs=1):
    """固定参照池（factor_lib 中全部因子）在某个切分上的五项指标，结果缓存到 CSV；n_jobs>1 时并行计算。"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"参照池_{split}_{label}.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0, dtype={0: str})
    panel_for(split)
    fns = {fid: fn for fid, _, _, fn, _ in FACTORS}
    out = _run_parallel(fns, [split], label, n_jobs)
    rows = {r["候选"]: {k: r[k] for k in METRICS} for r in out}
    t = pd.DataFrame(rows).T[METRICS]
    t.index = t.index.astype(str)
    t.to_csv(path)
    return t


def score_vs_pool(m, pool):
    """把单个候选放入固定参照池，计算五项百分位并平均（换手率越低越好）。"""
    t = pd.concat([pool, pd.DataFrame([m], index=["__cand__"])])[METRICS]
    r = {}
    for c in ["ic_mean", "ic_ir", "ls_sharpe", "stress"]:
        r[c] = t[c].rank(pct=True)["__cand__"]
    r["turnover"] = (-t["turnover"]).rank(pct=True)["__cand__"]
    return float(np.mean(list(r.values())))


_CANDS = {}


def _eval_one(args):
    name, split, label = args
    m = metrics_of(_CANDS[name](_PANELS[split]), split, label)
    return {"候选": name, "切分": split, **m}


def _run_parallel(fns, splits, label, n_jobs):
    """在已加载的面板上评估一组函数；n_jobs>1 时用 fork 多进程（子进程共享主进程内存中的面板）。"""
    _CANDS.clear()
    _CANDS.update(fns)
    tasks = [(name, split, label) for split in splits for name in fns]
    if n_jobs > 1 and len(tasks) > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(n_jobs, len(tasks))) as pool:
            return pool.map(_eval_one, tasks, chunksize=1)
    return [_eval_one(t) for t in tasks]


def evaluate_candidates(cands, splits=("开发", "验证"), label="y_off", n_jobs=1):
    """cands: {名称: 函数(面板)->Series}。返回每个候选在各切分上的原始指标与固定池综合分。

    n_jobs > 1 时用 fork 多进程并行（面板先在主进程加载，子进程共享内存），适合在远端大内存机器上运行。
    """
    for split in splits:
        panel_for(split)
    pools = {split: pool_metrics(split, label, n_jobs) for split in splits}
    out = _run_parallel(cands, splits, label, n_jobs)
    for row in out:
        row["综合分"] = score_vs_pool({k: row[k] for k in METRICS}, pools[row["切分"]])
    t = pd.DataFrame(out)
    wide = t.pivot(index="候选", columns="切分", values="综合分")
    wide.columns = [f"综合分_{c}" for c in wide.columns]
    ic = t.pivot(index="候选", columns="切分", values="ic_mean")
    ic.columns = [f"IC_{c}" for c in ic.columns]
    to = t.pivot(index="候选", columns="切分", values="turnover")
    to.columns = [f"换手_{c}" for c in to.columns]
    res = wide.join(ic).join(to)
    res["综合分_最差"] = wide.min(axis=1)
    return res.sort_values("综合分_最差", ascending=False), t
