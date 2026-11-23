# -*- coding: utf-8 -*-
"""评分与报告：读取 runner 产出的因子值，用常驻内存的评分核心计算全部指标，写 result.json 和 report.md。"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config as C  # noqa: E402
import engine  # noqa: E402

VERSION = "jyc-scorer-2.5.3"
SLOT_NAMES = {930: "09:30", 1000: "10:00", 1030: "10:30", 1100: "11:00", 1130: "11:30", 1330: "13:30", 1400: "14:00", 1430: "14:30"}
METRIC_CN = {"ic_mean": "IC 均值", "ic_ir": "IC_IR", "ls_sharpe": "多空夏普", "stress": "压力稳定性", "turnover": "换手率（越低越好）"}
OFFICIAL_PROBES = ["probe_label_2024Q1.parquet", "probe_label_202106.parquet"]


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return [clean(v) for v in x.tolist()]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return None if math.isnan(x) or math.isinf(x) else x
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def official_label(core):
    """平台导出的官方标签片段（2021-06、2024-01-02~03-01）对齐到评分核心；其余为 NaN。"""
    m = pd.read_csv(C.DATA / "exposure" / "id_map.csv", dtype=str)
    m = m[m["ok"].str.lower() == "true"]
    r2i = dict(zip(m["instrument"], m["local_id"]))
    y = np.full(core.n_rows, np.nan)
    for name in OFFICIAL_PROBES:
        p = C.DATA / "platform_probe" / name
        if not p.exists():
            continue
        lab = pd.read_parquet(p, columns=["date", "instrument", "vwap_return"])
        ids = lab["instrument"].map(r2i)
        ok = ids.notna()
        v, _ = core.align(lab.loc[ok, "date"].to_numpy("datetime64[ns]").astype(np.int64),
                          ids[ok].str[2:].astype(np.int64).to_numpy(), lab.loc[ok, "vwap_return"].to_numpy())
        y = np.where(np.isfinite(v), v, y)
    return y


def detail(core, s):
    ic = s["ic"]
    if len(ic) == 0:
        return {}
    med = np.median(s["ysd"])
    hi, lo = s["ysd"] > med, s["ysd"] <= med
    by_slot = {}
    for sl in np.unique(s["slot"]):
        x = ic[s["slot"] == sl]
        by_slot[SLOT_NAMES.get(int(sl), str(sl))] = {"ic_mean": float(x.mean()), "ic_ir": engine._ir(x), "n_xsec": int(len(x)),
                                                     "ic_positive_ratio": float((x > 0).mean())}
    lsv = s["ls"][np.isfinite(s["ls"])]
    return {"stress_high_dispersion_ic_ir": engine._ir(ic[hi]), "stress_low_dispersion_ic_ir": engine._ir(ic[lo]),
            "turnover_long": engine._nanmean(s["tl"]), "turnover_short": engine._nanmean(s["ts"]),
            "ls_mean_per_section": float(lsv.mean()) if len(lsv) else None, "ls_std_per_section": float(lsv.std(ddof=1)) if len(lsv) > 1 else None,
            "ic_std": float(ic.std(ddof=1)) if len(ic) > 1 else None, "ic_positive_ratio": float((ic > 0).mean()),
            "ic_quantiles": {str(q): float(np.quantile(ic, q)) for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)},
            "stocks_per_section_median": float(np.median(s["n"])), "by_slot": by_slot}


def periods(s, start, end):
    out_y, out_h = {}, {}
    days = pd.to_datetime(s["date"])
    for y in sorted(days.year.unique()):
        m = days.year == y
        out_y[str(y)] = engine.Core.summarize(s, np.asarray(m))
        for h, mm in ((1, days.month <= 6), (2, days.month > 6)):
            mk = np.asarray(m & mm)
            if mk.any():
                out_h[f"{y}H{h}"] = engine.Core.summarize(s, mk)
    return out_y, out_h


def coverage(core, f, dm):
    """因子在评估行上的覆盖：逐截面时段的有效比例、标准差、全零截面数（例如平台 SQL 的 09:30 缺陷）。"""
    rows = dm[core.date_id]
    out = {}
    for sl in np.unique(core.slot_of_date):
        r = rows & (core.slot_of_date[core.date_id] == sl)
        v = f[r]
        g = core.date_id[r]
        fin = np.isfinite(v)
        if not len(v):
            continue
        sd = pd.Series(np.where(fin, v, np.nan)).groupby(g).std(ddof=0)
        zero = pd.Series(np.where(fin, np.abs(v), np.nan)).groupby(g).max().fillna(0) == 0
        out[SLOT_NAMES.get(int(sl), str(sl))] = {"coverage": float(fin.mean()), "median_cross_sectional_std": float(sd.median()),
                                                "constant_or_all_zero_sections": int((sd.fillna(0) == 0).sum()), "all_zero_sections": int(zero.sum()),
                                                "sections": int(len(sd))}
    return out


def daily_curve(s):
    if len(s["ic"]) == 0:
        return []
    df = pd.DataFrame({"day": pd.to_datetime(s["day"]), "ic": s["ic"], "ls": s["ls"]})
    g = df.groupby("day").agg(ic=("ic", "mean"), ls=("ls", "mean"), n=("ic", "size"))
    g["cum_ic"] = g["ic"].cumsum()
    g["ic_ma20"] = g["ic"].rolling(20, min_periods=5).mean()
    return [{"day": d.strftime("%Y-%m-%d"), "ic": r.ic, "ic_ma20": r.ic_ma20, "cum_ic": r.cum_ic, "ls": r.ls} for d, r in g.iterrows()]


PAIRED_METRICS = ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]
MIN_PAIRED_DAYS = 60                         # 少于约 3 个月时自助法标准误不可靠


def _metrics_idx(ic, ls, ysd, tl, ts, idx):
    """在截面下标 idx（可重复，自助抽样用）上计算五项指标，口径同 engine.Core.summarize。"""
    x = ic[idx]
    sd = x.std(ddof=1)
    l = ls[idx]
    l = l[np.isfinite(l)]
    lsd = l.std(ddof=1) if len(l) > 1 else np.nan
    y = ysd[idx]
    med = np.median(y)
    hi, lo = x[y > med], x[y <= med]
    ir = lambda v: v.mean() / v.std(ddof=1) if len(v) > 1 and v.std(ddof=1) > 0 else np.nan
    with np.errstate(invalid="ignore"):
        return np.array([x.mean(), x.mean() / sd if sd > 0 else np.nan, l.mean() / lsd * engine.ANNUAL if lsd > 0 else np.nan,
                         min(ir(hi), ir(lo)), np.nanmean([np.nanmean(tl[idx]), np.nanmean(ts[idx])])])


def _pct_fns(ref_table):
    """按平台评分规则把五项指标换算成百分位：以参照池同区间的指标分布为总体，经验分布函数线性插值（换手越低越好）。"""
    fns = []
    for m in PAIRED_METRICS:
        v = np.sort(ref_table[m].dropna().to_numpy(np.float64))
        q = (np.arange(len(v)) + 0.5) / len(v)
        fns.append((lambda x, v=v, q=q: 1.0 - np.interp(x, v, q)) if m == "turnover" else (lambda x, v=v, q=q: np.interp(x, v, q)))
    return fns


def _composite(fns, mv):
    return float(np.mean([fn(x) for fn, x in zip(fns, mv)]))


def paired(core, s, sb, base_name, B=500, block_days=20, seed=0, ref_table=None):
    """候选与基线的配对比较：五项指标之差，以及按交易日分块自助法（块长 20 个交易日）的标准误。

    每个因子都用**自己实际参与评测的截面**计算指标（与平台一致：常数 / 全 0 截面被跳过，其余截面全部计入），
    自助法按交易日成块重抽样，同一批交易日上两个因子各取自己的截面。这样两边对 09:30 等截面的处理不同时，
    差异会如实反映在结果里（2026-10-12 以前的版本只比较两边共同的截面，会把这种差异漏掉）。
    同一批交易日上的市场噪声相互抵消，所以差值的标准误远小于单个指标的标准误。
    平台分数是五项指标百分位的等权平均，所以主结论看“综合百分位差”：五项各自换算成参照池中的百分位后等权平均，再相减。"""
    keys = ("ic", "ls", "ysd", "tl", "ts")
    A = [s[k] for k in keys]
    Bv = [sb[k] for k in keys]
    ud = np.union1d(s["day"], sb["day"])
    common_days = np.intersect1d(s["day"], sb["day"])
    if len(common_days) < 20 or len(s["ic"]) < 100 or len(sb["ic"]) < 100:
        return {"baseline_name": base_name, "error": f"与基线共同的交易日只有 {len(common_days)} 个"}

    def day_index(day):
        dpos = np.searchsorted(ud, day)
        order = np.argsort(dpos, kind="stable")
        st = np.searchsorted(dpos[order], np.arange(len(ud)))
        en = np.searchsorted(dpos[order], np.arange(len(ud)), side="right")
        return order, st, en

    oa, sa, ea = day_index(s["day"])
    ob, sb_, eb = day_index(sb["day"])
    block_days = int(min(block_days, max(2, len(ud) // 10)))      # 区间短时缩小块长，保证至少约 10 个块
    nb = int(np.ceil(len(ud) / block_days))
    ma, mb = _metrics_idx(*A, np.arange(len(A[0]))), _metrics_idx(*Bv, np.arange(len(Bv[0])))
    rng = np.random.default_rng(seed)
    fns = _pct_fns(ref_table) if ref_table is not None and len(ref_table) >= 5 else None
    cdiff = np.empty(B)
    diffs = np.empty((B, 5))
    for k in range(B):
        st = rng.integers(0, max(1, len(ud) - block_days + 1), nb)
        days = (st[:, None] + np.arange(block_days)).ravel()
        days = days[days < len(ud)]
        ia = np.concatenate([oa[sa[d]:ea[d]] for d in days])
        ib = np.concatenate([ob[sb_[d]:eb[d]] for d in days])
        xa, xb = _metrics_idx(*A, ia), _metrics_idx(*Bv, ib)
        diffs[k] = xa - xb
        if fns:
            cdiff[k] = _composite(fns, xa) - _composite(fns, xb)
    se = np.nanstd(diffs, axis=0, ddof=1)
    out = {}
    better, worse = [], []
    for j, m in enumerate(PAIRED_METRICS):
        d = float(ma[j] - mb[j])
        z = d / se[j] if se[j] > 0 else float("nan")
        good = -z if m == "turnover" else z                    # 换手越低越好
        out[m] = {"candidate": float(ma[j]), "baseline": float(mb[j]), "diff": d, "se": float(se[j]), "z": float(z)}
        if good >= 2:
            better.append(f"{METRIC_CN[m].split('（')[0]}（z={z:+.1f}）")
        elif good <= -2:
            worse.append(f"{METRIC_CN[m].split('（')[0]}（z={z:+.1f}）")
    comp = None
    capped = []
    if fns:
        ca, cb = _composite(fns, ma), _composite(fns, mb)
        for j, (m, fn) in enumerate(zip(PAIRED_METRICS, fns)):   # 基线在参照池里已是该项第一：候选再好也不加分
            top = max(fn(-1e18), fn(1e18))
            if fn(mb[j]) >= top - 1e-6 and fn(ma[j]) >= top - 1e-6:
                capped.append(METRIC_CN[m].split('（')[0])
        cse = float(np.nanstd(cdiff, ddof=1))
        cz = (ca - cb) / cse if cse > 0 else float("nan")
        comp = {"candidate": ca, "baseline": cb, "diff": ca - cb, "se": cse, "z": cz,
                "pct_diff": {m: float(fn(ma[j]) - fn(mb[j])) for j, (m, fn) in enumerate(zip(PAIRED_METRICS, fns))},
                "approx_score_diff": ca - cb, "capped_metrics": capped,
                "note": f"以参照池 {len(ref_table)} 个因子同区间的指标分布为总体换算百分位（平台的总体是全体参赛提交，头部更密，"
                        "实际分差可能更大或更小）；超出参照池范围的指标值按最高/最低百分位截断。"}
    daily_a = pd.Series(A[0]).groupby(s["day"]).mean()
    daily_b = pd.Series(Bv[0]).groupby(sb["day"]).mean()
    daily_a, daily_b = daily_a.align(daily_b, join="inner")
    slots_a = {SLOT_NAMES.get(int(x), str(x)): int(n) for x, n in zip(*np.unique(s["slot"], return_counts=True))}
    slots_b = {SLOT_NAMES.get(int(x), str(x)): int(n) for x, n in zip(*np.unique(sb["slot"], return_counts=True))}
    slot_note = None
    if set(slots_a) != set(slots_b):
        only_a, only_b = sorted(set(slots_a) - set(slots_b)), sorted(set(slots_b) - set(slots_a))
        slot_note = ("参与评测的截面时刻不同：" + (f"候选多出 {only_a}" if only_a else "") + ("，" if only_a and only_b else "")
                     + (f"基线多出 {only_b}" if only_b else "") + "（常数 / 全 0 截面被跳过）。差异已计入比较。")
    detail_v = ("显著更好：" + "、".join(better) if better else "") + ("；" if better and worse else "") + \
               ("显著更差：" + "、".join(worse) if worse else "")
    if comp:
        head = ("综合百分位显著更高" if comp["z"] >= 2 else "综合百分位显著更低" if comp["z"] <= -2 else "综合百分位没有显著差异") + \
               f"（{comp['diff']:+.3f} ± {comp['se']:.3f}，z={comp['z']:+.1f}）"
        verdict = head + ("。分项：" + detail_v if detail_v else "。五项各自都没有显著差异")
        if capped:
            verdict += f"。注意：{'、'.join(capped)}上两者在参照池里都已是第一，这几项的差异不计入综合百分位，请看分项 z"
    else:
        verdict = detail_v
    if slot_note:
        verdict = verdict + "。" + slot_note if verdict else slot_note
    if len(ud) < MIN_PAIRED_DAYS:
        verdict = f"区间只有 {len(ud)} 个交易日（少于 {MIN_PAIRED_DAYS}），标准误不可靠，不做显著性判断"
    return {"baseline_name": base_name, "n_sections_candidate": int(len(A[0])), "n_sections_baseline": int(len(Bv[0])),
            "n_common_sections": int(len(np.intersect1d(s["g"], sb["g"]))), "n_days": int(len(ud)),
            "sections_by_slot": {"candidate": slots_a, "baseline": slots_b}, "slot_note": slot_note,
            "bootstrap": {"replicates": B, "block_trading_days": block_days},
            "daily_ic_corr": float(np.corrcoef(daily_a, daily_b)[0, 1]),
            "diff": out, "composite": comp, "verdict": verdict or "与基线没有显著差异（五项指标的 |z| 都小于 2）",
            "note": f"diff = 候选 − 基线；每个因子用自己实际参与评测的截面（与平台一致）；se 为按 {block_days} 个交易日分块的自助法标准误；"
                    "|z| ≥ 2 视为显著。换手越低越好。"}


def _baseline_factor(core, refpool, spec):
    b = spec.get("baseline") or "ref:104"
    if b == "none":
        return None, None
    if b.startswith("ref:"):
        if refpool is None:
            return None, "参照池未载入，无法与 " + b + " 比较"
        v = refpool.values_for(b[4:])
        if v is None:
            return None, f"参照池里没有 {b[4:]} 号"
        src = "平台代码" if refpool.source == "platform" else "factor_lib 面板复刻"
        return (v, f"参照池 {b[4:]} 号（{refpool.info[b[4:]]['name']}，{src}）"), None
    p = C.JOBS / b / "factor_keys.parquet"
    if not p.exists():
        return None, f"基线任务 {b} 没有因子值"
    k = pd.read_parquet(p)
    fb, _ = core.align(k["key_date"].to_numpy("datetime64[ns]").astype(np.int64), k["instrument_id"].to_numpy(), k["factor"].to_numpy())
    st = C.JOBS / b / "state.json"
    nm = json.loads(st.read_text(encoding="utf-8")).get("name", b) if st.exists() else b
    return (fb, f"任务 {b}（{nm}）"), None


# 因子代码禁止读取的表：暴露表第 t 天的流通市值、风格按当天 15:00 收盘计算，ret 就是当天收益；vwap 表是标签用的未来窗口 VWAP。
# 二者都是评测设施，不是赛题给的行情数据，盘中截面用了就是未来数据。
FORBIDDEN_TABLES = {
    "cpt_jyc_2026_vwap": "VWAP 表（标签用的未来 30 分钟窗口 VWAP）",
}
# 2026-10-12 用户确认：因子赛道可以读暴露表，但第 t 天的市值/风格按当天 15:00 收盘计算、ret 是当天收益，
# 盘中截面只能用前一交易日及更早的值。读表本身不算违规；是否错位只能由代码保证，这里给出提示。
CAUTION_TABLES = {
    "cpt_jyc_2026_exposure": "暴露表：只能用前一交易日及更早的值（第 t 天的市值/风格按当天 15:00 收盘计算，ret 是当天收益），"
                             "盘中截面用当天的值就是未来数据",
}


def data_violations(job_dir: Path):
    """根据 DAI 模拟层记录的实际读表（dai_queries.json）判断是否读了禁用表。"""
    p = job_dir / "dai_queries.json"
    if not p.exists():
        return []
    try:
        tables = {t for q in json.loads(p.read_text(encoding="utf-8")) for t in q.get("tables", [])}
    except Exception:
        return []
    return [f"读取了 {t}：{why}" for t, why in FORBIDDEN_TABLES.items() if t in tables]


def data_notices(job_dir: Path):
    """读了需要错位使用的表（暴露表）时的提示，不影响排名。"""
    p = job_dir / "dai_queries.json"
    if not p.exists():
        return []
    try:
        tables = {t for q in json.loads(p.read_text(encoding="utf-8")) for t in (q.get("tables") or [])}
    except Exception:
        return []
    return [f"读取了 {t}：{why}" for t, why in CAUTION_TABLES.items() if t in tables]


def score(core, refpool, job_dir: Path, spec: dict, produced: dict):
    t0 = time.time()
    start = pd.Period(spec["start"], "M").start_time
    end = pd.Period(spec["end"], "M").end_time
    keys = pd.read_parquet(job_dir / "factor_keys.parquet")
    f, hit = core.align(keys["key_date"].to_numpy("datetime64[ns]").astype(np.int64), keys["instrument_id"].to_numpy(),
                        keys["factor"].to_numpy())
    dm = core.date_range_mask(start, end)
    eval_rows = int(dm[core.date_id].sum())
    in_range = int((np.isfinite(f) & dm[core.date_id]).sum())
    label = spec.get("label", "y_off")
    y = official_label(core) if label == "official" else None
    if y is not None:
        dm &= np.bincount(core.date_id, np.isfinite(y), core.n_dates) > 0
        if not dm.any():
            raise ValueError("所选区间没有官方标签（官方标签片段只有 2021-06 和 2024-01-02~03-01）")
    s = core.series(f, True, dm, y=y)
    m = core.summarize(s)
    if label == "y_off":                     # 赋分表从逐截面序列切出统一窗口的指标
        import leaderboard
        leaderboard.save_series(job_dir, s)
    s0 = core.series(f, False, dm, y=y)
    m0 = core.summarize(s0)
    by_year, by_half = periods(s, start, end)
    res = {
        "meta": {"version": VERSION, "job": job_dir.name, "kind": spec["kind"], "name": spec.get("name"), "inputs": spec.get("inputs"),
                 "start": str(start.date()), "end": str(end.date()), "label": label,
                 "label_note": "y_off = 本地由 1 分钟数据重建的官方标签（与官方截面秩相关约 0.97）" if label == "y_off"
                 else "官方标签（平台导出的片段，只覆盖 2021-06 与 2024-01-02~03-01）",
                 "produce_seconds": produced.get("produce_seconds"), "produce_peak_gb": produced.get("produce_peak_gb"),
                 "created": time.strftime("%Y-%m-%d %H:%M:%S")},
        "warnings": list(produced.get("warnings", [])),
        "platform_checks": produced.get("checks"),
        "mapping": produced.get("mapping"),
        "coverage": {"eval_rows_in_range": eval_rows, "factor_rows_aligned": int(hit), "factor_finite_rows_in_range": in_range,
                     "finite_coverage": in_range / eval_rows if eval_rows else None,
                     "official_exposure_row_coverage": float(core.has[dm[core.date_id]].mean()) if eval_rows else None,
                     "by_slot": coverage(core, f, dm)},
        "metrics": {k: m[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover", "n_xsec")},
        "detail": detail(core, s),
        "no_neutralize": {k: m0[k] for k in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")},
        "by_year": by_year, "by_half": by_half,
        "daily": daily_curve(s),
    }
    try:
        bf, berr = _baseline_factor(core, refpool, spec)
        if berr:
            res["warnings"].append(f"配对比较未进行：{berr}")
        elif bf is not None:
            sb = core.series(bf[0], True, dm, y=y)
            rt = None
            if refpool is not None:
                try:
                    rt = refpool.metrics(start, end).dropna()
                except Exception:
                    rt = None
            res["paired"] = paired(core, s, sb, bf[1], ref_table=rt)
    except Exception as e:
        res["warnings"].append(f"配对比较失败：{type(e).__name__}: {e}")
    if refpool is not None:
        try:
            pos = refpool.position(res["metrics"], start, end)
            if pos is None:
                res["warnings"].append("所选区间不在参照池覆盖范围内（参照池为平台代码在 2023-01 ~ 2024-12 上的结果），没有参照池百分位和估计分；"
                                       "配对比较请用任务号作基线，赋分表请用 2023–2024 窗口")
            else:
                res["reference_pool"] = pos
            res["correlations"] = refpool.correlations(core, f, start, end)
        except Exception as e:
            res["warnings"].append(f"参照池计算失败：{e}")
    w = res["warnings"]
    cov = res["coverage"]
    if cov["finite_coverage"] is not None and cov["finite_coverage"] < 0.6:
        w.append(f"因子只覆盖了 {cov['finite_coverage']:.0%} 的评估行；缺失的行按平台做法补 0 参与排名")
    for sl, c in cov["by_slot"].items():
        if c["sections"] and c["constant_or_all_zero_sections"] > 0.5 * c["sections"]:
            w.append(f"{sl} 截面有 {c['constant_or_all_zero_sections']}/{c['sections']} 个截面因子为常数或全 0（例如平台 SQL 的 09:30 缺陷）")
    pc = produced.get("checks") or {}
    if pc and not pc.get("pass", True):
        errs = []
        if pc.get("sections_over_40pct_missing"):
            worst = dict(list((pc.get("worst_sections") or {}).items())[:3])
            errs.append(f"v6.datachecker.DataValidationError: 截面因子缺失率不得高于 40%；详情={{'sample': {worst}}}"
                        f"（共 {pc['sections_over_40pct_missing']}/{pc.get('sections_total')} 个截面超标；缺失 = 股票池里没有有限因子值，NaN 和缺行都算，0 不算）")
        if pc.get("duplicate_keys"):
            errs.append(f"重复的 date + instrument：{pc['duplicate_keys']} 行")
        if pc.get("inf_count"):
            errs.append(f"含 inf：{pc['inf_count']} 个")
        if pc.get("rows_outside_8_slots"):
            errs.append(f"截面时刻不在 8 个约定时刻：{pc['rows_outside_8_slots']} 行")
        if not pc.get("exactly_three_columns", True):
            errs.append("输出不是恰好 date、instrument、factor 三列")
        res["platform_error"] = errs or ["输出不满足平台格式要求，见“平台规则检查”"]
        w.insert(0, "平台会报错、不给分（以下指标仅供诊断，不进赋分表）：" + "；".join(res["platform_error"]))
    dv = data_violations(job_dir)
    if dv:
        res["data_violation"] = dv
        w.insert(0, "数据违规（不进赋分表，不要提交）：" + "；".join(dv))
    dn = data_notices(job_dir)
    if dn:
        res["data_notice"] = dn
        w.append("数据提示：" + "；".join(dn))
    res["meta"]["score_seconds"] = round(time.time() - t0, 2)
    res = clean(res)
    (job_dir / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    (job_dir / "report.md").write_text(report_md(res), encoding="utf-8")
    return res


def _f(v, n=4):
    return "—" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def report_md(r):
    m, meta = r["metrics"], r["meta"]
    L = [f"# 评分报告：{meta.get('name') or meta['job']}", "",
         f"- 提交类型：{meta['kind']}；区间 {meta['start']} ~ {meta['end']}；标签：{meta['label_note']}",
         f"- 评分机 {meta['version']}；生成于 {meta['created']}；因子计算 {meta.get('produce_seconds')} s，评分 {meta.get('score_seconds')} s", ""]
    rp = r.get("reference_pool")
    head = (f"**结论**：有效截面 {m['n_xsec']} 个，IC {_f(m['ic_mean'])}，IC_IR {_f(m['ic_ir'], 3)}，多空夏普 {_f(m['ls_sharpe'], 2)}，"
            f"压力 {_f(m['stress'], 3)}，换手 {_f(m['turnover'], 3)}。")
    if rp and rp.get("estimated_public_score") is not None:
        head += f" 在本地参照池（{rp['n_refs']} 个有公榜分的因子）中的综合百分位 {rp['local_composite']:.2f}，粗略估计公榜分 {rp['estimated_public_score']:.3f}（仅供参考）。"
    L += [head, ""]
    if r.get("data_violation"):
        L += ["## ✗ 数据违规：不进赋分表，不要提交", ""] + [f"- {x}" for x in r["data_violation"]] + [""]
    if r.get("platform_error"):
        L += ["## ✗ 平台会报错、不给分", ""] + [f"- {x}" for x in r["platform_error"]] + ["", "以下指标仅供诊断。", ""]
    if r.get("warnings"):
        L += ["## ⚠ 提示", ""] + [f"- {x}" for x in r["warnings"]] + [""]
    L += ["## 五项原始指标（平台口径：风格 + 行业剔除）", "", "| 指标 | 值 | 不剔除风格（对照） |" + (" 参照池百分位 |" if rp else ""),
          "|---|---:|---:|" + ("---:|" if rp else "")]
    for k, cn in METRIC_CN.items():
        L.append(f"| {cn} | {_f(m[k])} | {_f(r['no_neutralize'][k])} |" + (f" {rp['percentile'][k]:.2f} |" if rp else ""))
    d = r.get("detail") or {}
    if d:
        L += ["", "## 分截面", "", "| 截面 | IC 均值 | IC_IR | 正 IC 比例 | 截面数 | 因子覆盖 | 常数/全 0 截面 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for sl, x in d["by_slot"].items():
            c = r["coverage"]["by_slot"].get(sl, {})
            L.append(f"| {sl} | {_f(x['ic_mean'])} | {_f(x['ic_ir'], 3)} | {_f(x['ic_positive_ratio'], 2)} | {x['n_xsec']} | "
                     f"{_f(c.get('coverage'), 3)} | {c.get('constant_or_all_zero_sections', '—')} |")
        L += ["", "## 压力、换手、IC 分布", "",
              f"- 压力：标签离散度高组 IC_IR {_f(d['stress_high_dispersion_ic_ir'], 3)}，低组 {_f(d['stress_low_dispersion_ic_ir'], 3)}（取较弱一组）",
              f"- 换手：多头 {_f(d['turnover_long'], 3)}，空头 {_f(d['turnover_short'], 3)}（只计同一交易日内相邻截面）",
              f"- 多空收益（每截面）均值 {_f(d['ls_mean_per_section'], 5)}，标准差 {_f(d['ls_std_per_section'], 5)}",
              f"- IC 标准差 {_f(d['ic_std'])}，正 IC 比例 {_f(d['ic_positive_ratio'], 3)}，每截面股票数中位 {_f(d['stocks_per_section_median'], 0)}",
              "- IC 分位数：" + "，".join(f"{q}: {_f(v)}" for q, v in d["ic_quantiles"].items())]
    L += ["", "## 分年 / 半年", "", "| 区间 | IC | IC_IR | 多空夏普 | 压力 | 换手 | 截面数 |", "|---|---:|---:|---:|---:|---:|---:|"]
    for p, x in {**r["by_year"], **r["by_half"]}.items():
        L.append(f"| {p} | {_f(x['ic_mean'])} | {_f(x['ic_ir'], 3)} | {_f(x['ls_sharpe'], 2)} | {_f(x['stress'], 3)} | {_f(x['turnover'], 3)} | {x['n_xsec']} |")
    pr = r.get("paired")
    if pr and not pr.get("error"):
        L += ["", f"## 与基线的配对比较：{pr['baseline_name']}", "",
              "平台分数 = 五项指标各自在全体提交中的百分位等权平均（各 20%，换手越低越好），所以先看“综合百分位”一行。", "",
              f"**{pr['verdict']}**。候选 {pr.get('n_sections_candidate')} 个截面、基线 {pr.get('n_sections_baseline')} 个（{pr['n_days']} 个交易日），日度 IC 相关 {_f(pr['daily_ic_corr'], 3)}。{pr['note']}", "",
              "| 指标 | 候选 | 基线 | 差 | 标准误 | z | 百分位差 |", "|---|---:|---:|---:|---:|---:|---:|"]
        cp = pr.get("composite") or {}
        if cp:
            L.append(f"| **综合百分位（五项等权，按评分规则）** | {_f(cp['candidate'], 3)} | {_f(cp['baseline'], 3)} | **{cp['diff']:+.3f}** | {_f(cp['se'], 3)} | {_f(cp['z'], 2)} | |")
        for k, cn in METRIC_CN.items():
            x = pr["diff"][k]
            pd_ = (cp.get("pct_diff") or {}).get(k)
            L.append(f"| {cn} | {_f(x['candidate'])} | {_f(x['baseline'])} | {_f(x['diff'])} | {_f(x['se'])} | {_f(x['z'], 2)} | {'—' if pd_ is None else f'{pd_:+.3f}'} |")
        if cp:
            L += ["", cp["note"]]
    elif pr:
        L += ["", f"配对比较：{pr['error']}"]
    if rp:
        L += ["", "## 参照池对比", "", rp["note"], "",
              f"参照池本地综合分与公榜分的秩相关 {_f(rp['calibration_rank_corr'], 3)}（越高说明本地排名越可信）。", "",
              "| 编号 | 因子 | 公榜分 | IC | IC_IR | 多空夏普 | 压力 | 换手 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
        for x in rp["top_refs"]:
            L.append(f"| {x['fid']} | {x['name']} | {_f(x['public_score'], 5)} | {_f(x['ic_mean'])} | {_f(x['ic_ir'], 3)} | {_f(x['ls_sharpe'], 2)} | {_f(x['stress'], 3)} | {_f(x['turnover'], 3)} |")
    if r.get("correlations"):
        L += ["", "与参照因子的截面秩相关（原始值）：" + "；".join(f"{k} {v['name']} {_f(v['rank_corr_mean'], 3)}" for k, v in r["correlations"].items())]
    pc = r.get("platform_checks") or {}
    if pc:
        L += ["", "## 平台规则检查", ""]
        for k, v in pc.items():
            L.append(f"- {k}：`{json.dumps(v, ensure_ascii=False)}`")
    mp = r.get("mapping") or {}
    if mp:
        L += ["", f"代码映射：真实代码 {mp.get('real_code_rows')} 行，伪代码 {mp.get('pseudo_code_rows')} 行，映射不上 {mp.get('unmapped_rows')} 行 {mp.get('unmapped_examples', '')}"]
    cv = r["coverage"]
    L += ["", f"覆盖：区间内评估行 {cv['eval_rows_in_range']:,}，因子有效 {cv['factor_finite_rows_in_range']:,}（{_f(cv['finite_coverage'], 3)}）；"
              f"有官方暴露的行 {_f(cv['official_exposure_row_coverage'], 3)}（没有暴露的股票不参与评估，与平台一致）。", ""]
    return "\n".join(L)
