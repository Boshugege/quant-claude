# -*- coding: utf-8 -*-
"""评分引擎：与 localeval.evaluate 同一口径，但把“与因子无关的部分”预先算好，单个因子 5 年全量评分只需几秒。

localeval.evaluate 每次都要逐截面解一次 OLS（风格 + 行业），并用 pandas groupby 计算秩、换手。这里拆成两部分：
- 预先计算（`build_core`，结果存在 NVMe 上，内存映射读取）：
  评估行（8 个截面）的日期、股票、标签、是否有官方暴露；每个截面 [1, 10 风格, 32 行业] 的列空间正交基 U
  （用 SVD 取秩，阈值与 np.linalg.lstsq 的默认 rcond 相同，所以 z − U·Uᵀz 就是 lstsq 的残差）。
- 每个因子（`Core.evaluate`）：inf/缺失补 0 → 截面标准化 → z − U·Uᵀz → 向量化的截面秩 → IC、多空、换手、压力。

与 localeval.evaluate 逐项一致（验收脚本比较五项指标和逐截面 IC，差异 < 1e-9）。
period（按年、半年）指标由逐截面序列直接切片得到，与对子区间单独运行 localeval.evaluate 的结果相同：
各截面的残差和秩互不影响，换手只在同一交易日内相邻截面之间计算，区间边界都落在交易日之间。
"""
from __future__ import annotations

import json
import math
import os
import time
import hashlib
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

N_SLOTS = 8
ANNUAL = math.sqrt(242 * N_SLOTS)
CORE_VERSION = 1


# ---------------------------------------------------------------- 构建（一次性）
def build_core(panel: pd.DataFrame, neutral_cols, out_dir: str | Path, workers: int = 8, label: str = "y_off"):
    """由 localeval.build_panel 的面板构建评分核心并写入 out_dir。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ev = panel["is_eval"].to_numpy()
    p = panel.loc[ev, ["key_date", "day", "slot", "instrument_id", label]].copy()
    B = panel.loc[ev, list(neutral_cols)].to_numpy(np.float64)
    p["has"] = np.isfinite(B).all(axis=1)
    order = np.lexsort((p["instrument_id"].to_numpy(), p["key_date"].to_numpy()))
    p = p.iloc[order].reset_index(drop=True)
    B = B[order]
    kd = p["key_date"].to_numpy("datetime64[ns]").astype(np.int64)
    dates, date_id = np.unique(kd, return_inverse=True)
    n_rows, n_dates = len(p), len(dates)
    day_of_date = p.groupby(date_id)["day"].first().to_numpy("datetime64[ns]").astype(np.int64)
    slot_of_date = p.groupby(date_id)["slot"].first().to_numpy().astype(np.int16)

    # 每个截面中有暴露的行（localeval：ev = is_eval & has；少于 50 行的截面不评估）
    has = p["has"].to_numpy()
    has_idx = np.flatnonzero(has)
    g_of = date_id[has_idx]
    starts = np.searchsorted(g_of, np.arange(n_dates))
    ends = np.searchsorted(g_of, np.arange(n_dates), side="right")
    cnt = ends - starts

    def basis(g):
        n = cnt[g]
        if n < 50:
            return None
        ii = has_idx[starts[g]:ends[g]]
        X = np.column_stack([np.ones(n), B[ii]])
        U, s, _ = np.linalg.svd(X, full_matrices=False)
        tol = np.finfo(np.float64).eps * max(X.shape) * s[0]     # 与 lstsq(rcond=None) 相同的秩判定
        return U[:, s > tol]

    with ThreadPoolExecutor(workers) as ex:
        bases = list(ex.map(basis, range(n_dates)))
    rank = np.array([0 if b is None else b.shape[1] for b in bases], np.int32)
    uoff = np.zeros(n_dates + 1, np.int64)
    uoff[1:] = np.cumsum(cnt * rank)
    U = np.lib.format.open_memmap(out / "U.npy", mode="w+", dtype=np.float64, shape=(int(uoff[-1]),))
    for g, b in enumerate(bases):
        if b is not None:
            U[uoff[g]:uoff[g + 1]] = b.ravel()
    U.flush()
    del U, bases
    np.savez(out / "core.npz", key_date=kd, date_id=date_id.astype(np.int32), dates=dates,
             day_of_date=day_of_date, slot_of_date=slot_of_date,
             inst=p["instrument_id"].to_numpy().astype(np.int32), y=p[label].to_numpy(np.float64), has=has,
             has_idx=has_idx.astype(np.int64), hstart=starts.astype(np.int64), hcnt=cnt.astype(np.int64),
             rank=rank, uoff=uoff)
    meta = {"version": CORE_VERSION, "rows": int(n_rows), "dates": int(n_dates), "label": label,
            "first": str(pd.Timestamp(dates[0])), "last": str(pd.Timestamp(dates[-1])),
            "exposure_row_coverage": float(has.mean()), "U_bytes": int(uoff[-1] * 8),
            "neutral_cols": list(neutral_cols), "build_seconds": round(time.time() - t0, 1)}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


# ---------------------------------------------------------------- 工具函数
def _group_rank_pct(gid, vals, n_groups):
    """按组的平均秩百分位（与 pandas groupby.rank(pct=True, method='average') 相同）。gid 须非降序。

    先按值全局快排，再按组做稳定排序（组号转 uint16 后 numpy 走基数排序），比两键 lexsort 快约 1.5 倍；
    并列值取平均秩，结果与排序方式无关，逐位相同。"""
    n = len(vals)
    if n == 0:
        return np.empty(0), np.zeros(n_groups, np.int64)
    o1 = np.argsort(vals, kind="quicksort")
    gg = gid[o1]
    if n_groups <= 65535:
        gg = gg.astype(np.uint16)
    order = o1[np.argsort(gg, kind="stable")]
    g, v = gid[order], vals[order]
    cnt = np.bincount(g, minlength=n_groups)
    gstart = np.r_[0, np.cumsum(cnt)[:-1]]
    change = np.empty(n, bool)
    change[0] = True
    change[1:] = (g[1:] != g[:-1]) | (v[1:] != v[:-1])
    first = np.flatnonzero(change)
    last = np.r_[first[1:] - 1, n - 1]
    pos_first = first - gstart[g[first]]
    pos_last = last - gstart[g[first]]
    run = np.cumsum(change) - 1
    r = ((pos_first + pos_last) / 2.0 + 1.0)[run]
    out = np.empty(n)
    out[order] = r / cnt[g]
    return out, cnt


def _digest(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(memoryview(a).cast("B"))
    return h.digest()


def _ir(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float("nan")
    s = x.std(ddof=1)
    return float(x.mean() / s) if s > 0 else float("nan")


def _nanmean(x):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if len(x) else float("nan")


# ---------------------------------------------------------------- 评分核心
class Core:
    def __init__(self, cache_dir: str | Path, workers: int = 8):
        d = Path(cache_dir)
        self.meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        z = np.load(d / "core.npz")
        for k in z.files:
            setattr(self, k, z[k])
        self.U = np.load(d / "U.npy", mmap_mode="r")
        self.n_rows, self.n_dates = len(self.key_date), len(self.dates)
        self.date_start = np.searchsorted(self.date_id, np.arange(self.n_dates))
        self.date_end = np.searchsorted(self.date_id, np.arange(self.n_dates), side="right")
        self.workers = workers
        self._row_index = None
        self._rank_cache = OrderedDict()          # 按内容寻址的秩缓存：标签、参照因子、候选因子的秩在多次调用和多个任务间复用
        self._rank_lock = threading.Lock()

    RANK_CACHE_BYTES = 768 * 2**20                # 秩缓存上限 768 MB（两年区间每项约 30 MB，五年约 80 MB）

    def _rank(self, sel, vals):
        """_group_rank_pct(date_id[sel], vals, n_dates)，以 (sel, vals) 的内容摘要缓存。"""
        key = _digest(sel, vals)
        with self._rank_lock:
            hit = self._rank_cache.get(key)
            if hit is not None:
                self._rank_cache.move_to_end(key)
                return hit
        out = _group_rank_pct(self.date_id[sel], vals, self.n_dates)
        for a in out:
            a.flags.writeable = False             # 缓存共享，禁止调用方原地修改
        with self._rank_lock:
            self._rank_cache[key] = out
            while len(self._rank_cache) > 1 and \
                    sum(x.nbytes + c.nbytes for x, c in self._rank_cache.values()) > self.RANK_CACHE_BYTES:
                self._rank_cache.popitem(last=False)
        return out

    # ---- 因子值对齐：(截面时间, 伪代码数字) -> 评估行
    def row_index(self):
        if self._row_index is None:
            self._row_index = pd.Index(self.date_id.astype(np.int64) * 100000 + self.inst.astype(np.int64))
        return self._row_index

    def align(self, key_date_ns, inst_id, values):
        """把 (截面时间 ns, 伪代码数字, 值) 对齐到评估行；返回对齐后的数组（缺失为 NaN）和命中行数。"""
        kd = np.asarray(key_date_ns, np.int64)
        did = np.clip(np.searchsorted(self.dates, kd), 0, self.n_dates - 1)
        hit = self.dates[did] == kd
        key = np.where(hit, did * 100000 + np.asarray(inst_id, np.int64), -1)
        pos = self.row_index().get_indexer(key)
        out = np.full(self.n_rows, np.nan)
        ok = pos >= 0
        out[pos[ok]] = np.asarray(values, np.float64)[ok]
        return out, int(ok.sum())

    def date_range_mask(self, start=None, end=None):
        m = np.ones(self.n_dates, bool)
        if start is not None:
            m &= self.dates >= pd.Timestamp(start).value
        if end is not None:
            m &= self.dates <= pd.Timestamp(end).value
        return m

    # ---- 风格 + 行业剔除
    def residual(self, f, date_mask=None):
        v = np.where(np.isfinite(f), f, 0.0)          # localeval：inf → NaN → 补 0
        out = np.full(self.n_rows, np.nan)
        U, has_idx, hs, hc, rk, uo = self.U, self.has_idx, self.hstart, self.hcnt, self.rank, self.uoff
        todo = np.flatnonzero((hc >= 50) & (rk > 0) & (np.ones(self.n_dates, bool) if date_mask is None else date_mask))

        def work(chunk):
            for g in chunk:
                ii = has_idx[hs[g]:hs[g] + hc[g]]
                x = v[ii]
                sd = x.std()
                if not sd > 0:
                    continue
                z = (x - x.mean()) / sd
                Ug = U[uo[g]:uo[g + 1]].reshape(hc[g], rk[g])
                out[ii] = z - Ug @ (Ug.T @ z)

        chunks = np.array_split(todo, max(1, min(self.workers * 4, len(todo))))
        if self.workers > 1 and len(todo) > 200:
            with ThreadPoolExecutor(self.workers) as ex:
                list(ex.map(work, chunks))
        else:
            work(todo)
        return out

    # ---- 逐截面序列
    def series(self, f, neutralize=True, date_mask=None, min_stocks=100, y=None):
        """返回逐截面的 IC、多空收益、标签离散度、两腿换手等（只含 n >= min_stocks 的截面，与 localeval 相同）。"""
        if neutralize:
            res = self.residual(f, date_mask)
        else:
            res = np.where(np.isfinite(f), f, 0.0).astype(np.float64)
            if date_mask is not None:
                res = np.where(date_mask[self.date_id], res, np.nan)
        y = self.y if y is None else y
        sel = np.flatnonzero(np.isfinite(res) & np.isfinite(y))
        g = self.date_id[sel]
        D = self.n_dates
        rf, n = self._rank(sel, res[sel])
        ry, _ = self._rank(sel, y[sel])
        nz = np.maximum(n, 1)
        xf = rf - (np.bincount(g, rf, D) / nz)[g]
        yf = ry - (np.bincount(g, ry, D) / nz)[g]
        xy, xx, yy = np.bincount(g, xf * yf, D), np.bincount(g, xf * xf, D), np.bincount(g, yf * yf, D)
        with np.errstate(invalid="ignore", divide="ignore"):
            ic = xy / np.sqrt(xx * yy)
        ok = (n >= min_stocks) & np.isfinite(ic)
        ysel = y[sel]
        top, bot = rf > 0.9, rf <= 0.1
        with np.errstate(invalid="ignore", divide="ignore"):
            ls = np.bincount(g, ysel * top, D) / np.bincount(g, top, D) - np.bincount(g, ysel * bot, D) / np.bincount(g, bot, D)
            ym = np.bincount(g, ysel, D) / nz
            ysd = np.sqrt(np.bincount(g, (ysel - ym[g]) ** 2, D) / (n - 1))
        kept = np.flatnonzero(ok)                    # 与 localeval 的 ic.index 相同
        tl = self._turnover(kept, g, top, sel)
        ts = self._turnover(kept, g, bot, sel)
        return {"g": kept, "date": self.dates[kept], "day": self.day_of_date[kept], "slot": self.slot_of_date[kept],
                "ic": ic[kept], "ls": ls[kept], "ysd": ysd[kept], "n": n[kept], "tl": tl, "ts": ts,
                "_res": res, "_sel": sel, "_rf": rf}

    def _turnover(self, kept, g, mask, sel):
        """leg 在相邻保留截面（同一交易日）之间的替换比例；不同交易日或成分为空记 NaN。"""
        K = len(kept)
        out = np.full(K, np.nan)
        if K < 2:
            return out
        pos = np.full(self.n_dates, -1, np.int64)
        pos[kept] = np.arange(K)
        k = pos[g[mask]]
        inst = self.inst[sel[mask]].astype(np.int64)
        good = k >= 0
        k, inst = k[good], inst[good]
        size = np.bincount(k, minlength=K)
        cur = np.unique(k * 100000 + inst)
        prev_shift = np.unique((k + 1) * 100000 + inst)       # 上一个保留截面的成分，挪到下一个位置
        both = np.intersect1d(cur, prev_shift, assume_unique=True)
        overlap = np.bincount(both // 100000, minlength=K)
        day = self.day_of_date[kept]
        same = np.r_[False, day[1:] == day[:-1]]
        with np.errstate(invalid="ignore", divide="ignore"):
            t = 1.0 - overlap / np.where(size > 0, size, np.nan)
        out[same] = t[same]
        return out

    # ---- 汇总
    @staticmethod
    def summarize(s, mask=None):
        """由逐截面序列（可按截面子集 mask）计算五项指标，口径同 localeval.evaluate。"""
        if mask is None:
            mask = np.ones(len(s["ic"]), bool)
        ic, ls, ysd = s["ic"][mask], s["ls"][mask], s["ysd"][mask]
        if len(ic) == 0:
            return {"ic_mean": float("nan"), "ic_ir": float("nan"), "ls_sharpe": float("nan"), "stress": float("nan"),
                    "turnover": float("nan"), "n_xsec": 0, "slots": {}}
        lsv = ls[np.isfinite(ls)]
        sd = lsv.std(ddof=1) if len(lsv) > 1 else float("nan")
        med = np.median(ysd)
        stress = min(_ir(ic[ysd > med]), _ir(ic[ysd <= med]))
        slots = {int(sl): float(ic[s["slot"][mask] == sl].mean()) for sl in np.unique(s["slot"][mask])}
        return {"ic_mean": float(ic.mean()), "ic_ir": _ir(ic),
                "ls_sharpe": float(lsv.mean() / sd * ANNUAL) if sd > 0 else float("nan"),
                "stress": stress, "turnover": float(np.nanmean([_nanmean(s["tl"][mask]), _nanmean(s["ts"][mask])])),
                "n_xsec": int(len(ic)), "slots": slots}

    def evaluate(self, f, start=None, end=None, neutralize=True, y=None):
        """五项指标（与 localeval.evaluate 对 [start, end] 区间的结果相同）。"""
        dm = self.date_range_mask(start, end)
        s = self.series(f, neutralize, dm, y=y)
        return self.summarize(s), s

    def per_date_rank_corr(self, a, b, date_mask=None):
        """两个因子（原始值）逐截面的 Spearman 秩相关均值（各截面去掉任一方缺失的行）。"""
        ok = np.isfinite(a) & np.isfinite(b)
        if date_mask is not None:
            ok &= date_mask[self.date_id]
        sel = np.flatnonzero(ok)
        g = self.date_id[sel]
        D = self.n_dates
        ra, n = self._rank(sel, a[sel])
        rb, _ = self._rank(sel, b[sel])
        nz = np.maximum(n, 1)
        xa = ra - (np.bincount(g, ra, D) / nz)[g]
        xb = rb - (np.bincount(g, rb, D) / nz)[g]
        with np.errstate(invalid="ignore", divide="ignore"):
            c = np.bincount(g, xa * xb, D) / np.sqrt(np.bincount(g, xa * xa, D) * np.bincount(g, xb * xb, D))
        c = c[(n >= 3) & np.isfinite(c)]
        return float(c.mean()) if len(c) else float("nan")
