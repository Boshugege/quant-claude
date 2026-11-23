# -*- coding: utf-8 -*-
"""新根因子的离线评估工具（任务书 §2 第 1–3 级口径）。

功能：
  --ic   <任务名>[...]      离线逐截面 IC（对齐评测机口径；与 result.json 的 ic_mean 对照，用于口径校准）
  --res  <任务名> [--h H]   残差 IC：截面内把 rank(X) 对 rank(H) 做 OLS 取残差 R，再算 Spearman(R, y)（第 2 级）
  --grid <任务名> [--h H]   离线合成网格 B_w = rank(H) + w·rank(X)，给出逐截面 IC 差（配对 z）与近似换手（第 3 级）
  --slot <任务名>           分时段 IC（第 0 级的分时段签名检查）
参数：--start/--end（默认 2023-01~2024-12）、--h（默认 fm_490_indneu25）、--ws（默认 0.10,0.20,0.35,0.50）
一律剔除 09:30 截面；一律用 /home/user/jyc_scorer/.venv/bin/python 运行。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

SCORER = pathlib.Path("/home/user/jyc_scorer")
sys.path.insert(0, str(SCORER))
import config as C  # noqa: E402  顶层 config（build/ 下另有一份，必须先导入顶层）
import engine  # noqa: E402

JOBS = [pathlib.Path("/home/user/jyc_scorer/jobs"), pathlib.Path("/srv/media/jyc_scorer/jobs_archive")]
RES = pathlib.Path("/home/user/quant-claude/工具/评分机/结果")
MIN_STOCKS = 100
_CORE = None


def core():
    global _CORE
    if _CORE is None:
        _CORE = engine.Core(str(C.CORE_DIR))
    return _CORE


_JOBDIR = {}


def jobdir(name):
    if name in _JOBDIR:
        return _JOBDIR[name]
    d = _jobdir_scan(name)
    _JOBDIR[name] = d
    return d


def _jobdir_scan(name):
    for base in JOBS:                     # values 任务：目录名就是任务号，没有 spec.json
        d = base / name
        if (d / "factor_keys.parquet").exists():
            return d
    for base in JOBS:
        for p in sorted(base.glob("*/spec.json")):
            try:
                sp = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if sp.get("name") != name:
                continue
            d = p.parent
            if (d / "factor_keys.parquet").exists():
                return d
    raise SystemExit("找不到任务名 %s（需要有 factor_keys.parquet）" % name)


def load(name):
    d = jobdir(name)
    keys = pd.read_parquet(d / "factor_keys.parquet")
    f, hit = core().align(keys["key_date"].to_numpy("datetime64[ns]").astype(np.int64),
                          keys["instrument_id"].to_numpy(), keys["factor"].to_numpy())
    return f, hit, d


def base_rows(start, end):
    """与评测机一致的区间语义：start 取当日 00:00，end 取当日 23:59:59.999999999（否则最后一天的盘中截面会被切掉）。"""
    c = core()
    end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    dm = c.date_range_mask(pd.Timestamp(start).normalize(), end_ts)
    did = c.date_id
    slot = c.slot_of_date[did]
    ok = np.isfinite(c.y) & dm[did] & (slot != 930)
    return ok, did, slot


def _sec_stats(df, cols):
    """逐截面相关：返回 cols[0] 与 cols[1:] 各列的皮尔逊相关（对秩即为 Spearman）。"""
    g = df.groupby("did")
    r0 = g[cols[0]].rank(pct=True)
    d0 = r0 - r0.groupby(df["did"]).transform("mean")
    n = g.size()
    out = {}
    for c in cols[1:]:
        rc = g[c].rank(pct=True)
        dc = rc - rc.groupby(df["did"]).transform("mean")
        out[c] = (d0 * dc).groupby(df["did"]).sum() / np.sqrt(
            (d0 * d0).groupby(df["did"]).sum() * (dc * dc).groupby(df["did"]).sum())
    return out, n


def section_corr(df, a, b):
    g = df.groupby("did")
    ra = g[a].rank(pct=True)
    rb = g[b].rank(pct=True)
    da = ra - ra.groupby(df["did"]).transform("mean")
    db = rb - rb.groupby(df["did"]).transform("mean")
    c = (da * db).groupby(df["did"]).sum() / np.sqrt(
        (da * da).groupby(df["did"]).sum() * (db * db).groupby(df["did"]).sum())
    n = g.size()
    return c[(n >= MIN_STOCKS) & c.notna()]


def nw_t(x, lag=5):
    x = np.asarray(pd.Series(x).dropna(), float)
    T = len(x)
    if T < 10:
        return float("nan"), float("nan"), T
    m = x.mean()
    d = x - m
    g0 = (d * d).mean()
    s = g0
    for l in range(1, lag + 1):
        gl = (d[l:] * d[:-l]).mean()
        s += 2.0 * (1.0 - l / (lag + 1.0)) * gl
    se = np.sqrt(max(s, 1e-30) / T)
    return m, (m / se if se > 0 else float("nan")), T


def year_of(idx):
    try:
        return pd.to_datetime(idx).year
    except Exception:
        return None


def ic_of(x, residual_on=None, start="2023-01-01", end="2024-12-31"):
    """逐截面 IC（Spearman）。residual_on 非空时，先逐截面把 rank(x) 对 rank(residual_on) 做 OLS 取残差，再算 IC。"""
    c = core()
    ok, did, slot = base_rows(start, end)
    rows = ok & np.isfinite(x)
    if residual_on is not None:
        rows &= np.isfinite(residual_on)
    idx = np.flatnonzero(rows)
    df = pd.DataFrame({"did": did[idx], "y": c.y[idx], "x": np.asarray(x[idx], float)})
    if residual_on is not None:
        df["h"] = np.asarray(residual_on[idx], float)
        g = df.groupby("did")
        rx = g["x"].rank(pct=True) - 0.5
        rh = g["h"].rank(pct=True) - 0.5
        mx = rx.groupby(df["did"]).transform("mean")
        mh = rh.groupby(df["did"]).transform("mean")
        xc, hc = rx - mx, rh - mh
        num = (hc * xc).groupby(df["did"]).sum()
        den = (hc * hc).groupby(df["did"]).sum().replace(0, np.nan)
        beta = (num / den).reindex(df["did"]).to_numpy()
        df["x"] = xc.to_numpy() - beta * hc.to_numpy()
    ic, n = _sec_stats(df, ["x", "y"])
    s = ic["y"][(n >= MIN_STOCKS) & ic["y"].notna()]
    return s


def daily_series(s, days):
    """把逐截面 IC 按交易日平均，得到逐日 IC 序列（任务书 §2 的 t 值口径：按日平均后跨日计算）。"""
    df = pd.DataFrame({"ic": s.to_numpy(float), "day": days})
    g = df.groupby("day")["ic"].mean()
    return g.sort_index()


def show_ic(s, label=""):
    if len(s) == 0:      # 退化情形：残差恒为 0（例如 X = H），相关无定义，按 0 记
        return {"label": label, "mean": 0.0, "t": float("nan"), "n_xsec": 0, "t_day": float("nan"), "n_day": 0,
                "by_year": {}, "by_year_day": {}}, pd.DatetimeIndex([])
    m, t, T = nw_t(s)
    days = pd.to_datetime([pd.Timestamp(int(core().dates[int(d)])) for d in s.index]).normalize()
    out = {"label": label, "mean": m, "t": t, "n_xsec": T}
    daily = daily_series(s, days)
    m1, t1, T1 = nw_t(daily)
    out["t_day"], out["n_day"], out["mean_day"] = t1, T1, m1
    by_year, by_year_day = {}, {}
    for y in sorted(set(days.year)):
        sel = s[[i for i, dd in zip(s.index, days) if dd.year == y]]
        m2, t2, T2 = nw_t(sel)
        by_year[int(y)] = {"mean": m2, "t": t2, "n": T2}
        if (daily.index.year == y).any():
            m3, t3, T3 = nw_t(daily[daily.index.year == y])
            by_year_day[int(y)] = {"mean": m3, "t": t3, "n": T3}
    out["by_year"] = by_year
    out["by_year_day"] = by_year_day
    return out, days


def fmt(d):
    s = "%sIC %.5f (t截面=%+.1f, n=%d; t日=%+.1f, n日=%d)" % (
        d["label"], d["mean"], d["t"], d["n_xsec"], d.get("t_day", float("nan")), d.get("n_day", 0))
    for y, v in d["by_year"].items():
        s += " | %d %.5f" % (y, v["mean"])
    return s


def platform_res(f):
    """平台口径预处理：风格 + 行业剔除（与评测机 core.residual 同一实现）。"""
    return core().residual(f)


def cmd_ic(args):
    for name in args.names:
        x, hit, d = load(name)
        s = ic_of(platform_res(x), start=args.start, end=args.end)
        out, _ = show_ic(s)
        res = RES / d.name / "result.json"
        pub = ""
        if res.exists():
            rj = json.load(open(res, encoding="utf-8"))
            pub = " | result.json ic_mean=%.5f (差 %+.5f)" % (rj["metrics"]["ic_mean"], out["mean"] - rj["metrics"]["ic_mean"])
        print("%-20s %s%s" % (name, fmt(out), pub))


def cmd_slot(args):
    for name in args.names:
        x, hit, d = load(name)
        x = platform_res(x)          # 与 result.json 的 detail.by_slot 同口径（风格+行业剔除）
        c = core()
        ok, did, slot = base_rows(args.start, args.end)
        rows = ok & np.isfinite(x)
        idx = np.flatnonzero(rows)
        df = pd.DataFrame({"did": did[idx], "slot": slot[idx], "y": c.y[idx], "x": x[idx]})
        ic, n = _sec_stats(df, ["x", "y"])
        keep = (n >= MIN_STOCKS) & ic["y"].notna()
        tab = pd.DataFrame({"ic": ic["y"][keep], "slot": df.groupby("did")["slot"].first()[keep]}).groupby("slot")["ic"].mean()
        med = float(np.median(tab.values))
        print("%-20s 分时段 IC：%s | 中位 %.5f" % (name, "，".join("%d %.5f" % (k, v) for k, v in tab.items()), med))
        if "1000" in tab.index.astype(str) and False:
            pass
        a = float(tab.get(1000, np.nan))
        b = float(tab.get(1430, np.nan))
        other = tab.drop(index=[i for i in (1000, 1430) if i in tab.index])
        # 一律按 |IC| 比较（因子方向可能为负，直接用带符号值会把"更强的负 IC"误判成泄漏）
        o_med = float(np.median(np.abs(other))) if len(other) else float("nan")
        if len(other) and np.isfinite(o_med) and o_med > 0 and np.isfinite(a) and abs(a) > 1.5 * o_med:
            print("        ⚠ |10:00 IC| 高于其它时段 |中位数| 1.5 倍：疑似泄漏签名")
        if len(other) and np.isfinite(o_med) and o_med > 0 and np.isfinite(b) and abs(b) > 1.5 * o_med:
            print("        ⚠ |14:30 IC| 高于其它时段 |中位数| 1.5 倍：疑似泄漏签名")


def cmd_res(args):
    h, _, _ = load(args.h)
    for name in args.names:
        x, hit, d = load(name)
        s_res = ic_of(x, residual_on=h, start=args.start, end=args.end)
        out, days = show_ic(s_res, "残差")
        c = core()
        ok, did, slot = base_rows(args.start, args.end)
        rows = ok & np.isfinite(x) & np.isfinite(h)
        idx = np.flatnonzero(rows)
        df = pd.DataFrame({"did": did[idx], "x": x[idx], "h": h[idx]})
        corr = section_corr(df, "x", "h")
        print("%-20s %s | corr(X,H)=%.4f" % (name, fmt(out), corr.mean()))
        # 分年残差 IC
        by = {}
        for y in (2023, 2024):
            sel = [i for i, dd in zip(s_res.index, days) if dd.year == y]
            by[y] = nw_t(s_res[sel])[0]
        _byday = out.get("by_year_day", {})
        _ok_day = (_byday.get(2023, {}).get("mean", -1) > 0) and (_byday.get(2024, {}).get("mean", -1) > 0)
        print("        分年残差 IC（截面口径）：2023 %+.5f，2024 %+.5f；按日口径：2023 %+.5f，2024 %+.5f"
              % (by[2023], by[2024], _byday.get(2023, {}).get("mean", float("nan")),
                 _byday.get(2024, {}).get("mean", float("nan"))))
        print("        门槛（按日 t，任务书 §2 口径）残差 ≥0.010 且 t日≥3 且两年 >0 且 |corr|≤0.6 → %s（截面 t：%+.1f）"
              % ("PASS" if (out["mean"] >= 0.010 and out.get("t_day", 0) >= 3 and _ok_day and abs(corr.mean()) <= 0.6) else "未过",
                 out["t"]))


def approx_turnover(ranks, did, days, slot, inst):
    """近似换手：同一交易日内相邻截面的秩相关均值，取 1 − 均值（向量化，按日为块）。"""
    df = pd.DataFrame({"r": ranks, "day": days, "slot": slot, "inst": inst})
    df = df[df["slot"] != 930].dropna(subset=["r"])
    if not len(df):
        return float("nan")
    ORDER = {1000: 0, 1030: 1, 1100: 2, 1130: 3, 1330: 4, 1400: 5, 1430: 6}
    df = df[df["slot"].isin(ORDER)].copy()
    df["ord"] = df["slot"].map(ORDER)
    out = []
    for _, sub in df.groupby("day", sort=True):
        m = sub.pivot_table(index="inst", columns="ord", values="r")
        cols = sorted(c for c in m.columns if m[c].notna().sum() >= 20)
        if len(cols) < 2:
            continue
        R = m[cols].rank(pct=True).to_numpy(float)
        for a, b in zip(range(len(cols) - 1), range(1, len(cols))):
            ok = np.isfinite(R[:, a]) & np.isfinite(R[:, b])
            if ok.sum() < 20:
                continue
            ra = R[ok, a] - R[ok, a].mean()
            rb = R[ok, b] - R[ok, b].mean()
            den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
            if den > 0:
                out.append((ra * rb).sum() / den)
    return 1.0 - float(np.mean(out)) if out else float("nan")


TURN_CAL = None      # §1.2 换手校准系数 {a,b,r2,n}，由 因子实验/round2_calib_turnover.py 拟合后写入


def _turn_cal():
    global TURN_CAL
    if TURN_CAL is None:
        f = pathlib.Path(__file__).with_name("turnover_calib.json")
        TURN_CAL = json.load(open(f, encoding="utf-8")) if f.exists() else {}
    return TURN_CAL


def cmd_grid(args):
    """第 3 级：离线合成网格 B = H原始值 + w·rank(X)（剔除 09:30），报 IC 差（配对 z）、近似换手，
    并用 §1.2 校准过的系数给出平台 Δ换手的估计。`--ewma` 时先对 X 在同一股票序列上做 EWMA 再排名。"""
    h, _, _ = load(args.h)
    c = core()
    ok, did, slot = base_rows(args.start, args.end)
    print("== 合成网格（raw 口径，剔除 09:30；H=%s）" % args.h)
    tcal = _turn_cal()
    if tcal:
        print("   换手校准：平台Δ ≈ %.3f·离线Δ + %.4f（R²=%.3f, n=%d）" % (tcal["a"], tcal["b"], tcal["r2"], tcal["n"]))
    rows0 = ok & np.isfinite(h)
    idx0 = np.flatnonzero(rows0)
    base = pd.DataFrame({"did": did[idx0], "inst": c.inst[idx0], "h": h[idx0]})
    g = base.groupby("did")
    base["rh"] = (g["h"].rank(pct=True) - 0.5).to_numpy()
    days = pd.to_datetime([pd.Timestamp(int(c.dates[int(k)])) for k in did[idx0]]).normalize()
    slots = slot[idx0]
    rh = base["rh"].to_numpy()
    fh = np.full(c.n_rows, np.nan)
    fh[idx0] = rh
    ic_h = ic_of(fh, start=args.start, end=args.end)
    turn_h = approx_turnover(rh, did[idx0], days, slots, base["inst"].to_numpy())
    hr = h[idx0]
    turn_h_raw = approx_turnover(hr, did[idx0], days, slots, base["inst"].to_numpy())
    alphas = [None] if not args.ewma else [float(a) for a in args.ewma.split(",")]
    out = {}
    for name in args.names:
        x, hit, d = load(name)
        ws = [float(w) for w in args.ws.split(",")]
        for alpha in alphas:
            xe = x
            if args.res_ewma:
                # 先逐截面把 X 对 H 取残差（只留 H 没有的那部分），再做 EWMA。
                # 变量一律加 _ 前缀，避免与上面 idx0 口径的 rh 撞名（曾因此把网格打成全 NaN）。
                _tmp = pd.DataFrame({"did": did, "h": h, "x": x})
                _g = _tmp.groupby("did")
                _rx = _g["x"].rank(pct=True) - 0.5
                _rh = _g["h"].rank(pct=True) - 0.5
                _mx = _rx.groupby(_tmp["did"]).transform("mean")
                _mh = _rh.groupby(_tmp["did"]).transform("mean")
                _num = ((_rh - _mh) * (_rx - _mx)).groupby(_tmp["did"]).sum()
                _den = ((_rh - _mh) ** 2).groupby(_tmp["did"]).sum().replace(0, np.nan)
                _beta = (_num / _den).reindex(_tmp["did"]).to_numpy()
                xe = (_rx - _mx).to_numpy() - _beta * (_rh - _mh).to_numpy()
            xa = xe if alpha is None else ewma_along(xe, did, c.inst, alpha)
            rows1 = rows0 & np.isfinite(xa)
            idx1 = np.flatnonzero(rows1)
            b1 = pd.DataFrame({"did": did[idx1], "rx": xa[idx1]})
            g1 = b1.groupby("did")
            b1["rx"] = (g1["rx"].rank(pct=True) - 0.5).to_numpy()
            keep = np.isin(idx1, idx0)
            idx1, rx = idx1[keep], b1["rx"].to_numpy()[keep]
            rh1 = rh[np.searchsorted(idx0, idx1)]
            pre = "res+" if args.res_ewma else ""
            tag = name if alpha is None else "%s/%sewma%.2f" % (name, pre, alpha)
            turns, ics = {}, {}
            hr1 = hr[np.searchsorted(idx0, idx1)]
            for w in ws:
                if args.inter:
                    b = np.where(np.isfinite(hr1) & np.isfinite(rx), hr1 * (1.0 + w * rx), np.nan)
                else:
                    b = np.where(np.isfinite(rh1) & np.isfinite(rx), rh1 + w * rx, np.nan)
                fb = np.full(c.n_rows, np.nan)
                fb[idx1] = b
                ic_b = ic_of(fb, start=args.start, end=args.end)
                common = ic_b.index.intersection(ic_h.index)
                diff = ic_b[common] - ic_h[common]
                m, tt, T = nw_t(diff)
                dd = pd.to_datetime([pd.Timestamp(int(c.dates[int(k)])) for k in common]).normalize()
                yr = {}
                for y in (2023, 2024):
                    sel = diff[[v.year == y for v in dd]]
                    yr[y] = nw_t(sel)[0] if len(sel) else float("nan")
                ib = nw_t(ic_b[common])[0]
                tb_turn = approx_turnover(b, did[idx1], days[np.searchsorted(idx0, idx1)], slots[np.searchsorted(idx0, idx1)], c.inst[idx1])
                dtoff = tb_turn - turn_h
                pred = (tcal["a"] * (tb_turn - turn_h_raw) + tcal["b"]) if tcal else float("nan")
                gate = (tt >= 3 and yr[2023] > 0 and yr[2024] > 0 and dtoff <= 0.010)
                print("  %-16s w=%+.2f IC 差 %+.5f (z=%+.1f) 2023 %+.5f 2024 %+.5f | 近似换手 Δ%+.4f（预测平台Δ%+.4f）%s"
                      % (tag, w, m, tt, yr[2023], yr[2024], dtoff, pred, "[PASS]" if gate else ""))
                ics[w] = ib
                turns[w] = tb_turn
            out[tag] = (ics, turns)
    print("  （H 单独：IC %.5f，t=%+.1f，近似换手 %.4f）" % (ic_h.mean(), nw_t(ic_h)[1], turn_h))
    return out


def residuals_vs_h(name, h, ok, did):
    """返回 (面板行号, 截面 id, 残差) —— 把 rank(X) 对 rank(H) 逐截面回归后的残差。

    带上面板行号是为了让调用方能按行对齐做 merge；只用 did 对齐会因为 did 是交易日
    （每族上千行）而产生笛卡尔积，直接把内存打爆（见 §4.1）。"""
    x, hit, d = load(name)
    rows = ok & np.isfinite(x) & np.isfinite(h)
    idx = np.flatnonzero(rows)
    df = pd.DataFrame({"idx": idx, "did": did[idx], "h": h[idx], "x": x[idx]})
    g = df.groupby("did")
    rx = (g["x"].rank(pct=True) - 0.5)
    rh = (g["h"].rank(pct=True) - 0.5)
    mx = rx.groupby(df["did"]).transform("mean")
    mh = rh.groupby(df["did"]).transform("mean")
    xc, hc = rx - mx, rh - mh
    num = (hc * xc).groupby(df["did"]).sum()
    den = (hc * hc).groupby(df["did"]).sum().replace(0, np.nan)
    beta = (num / den).reindex(df["did"]).to_numpy()
    res = xc.to_numpy() - beta * hc.to_numpy()
    return df["idx"].to_numpy(), df["did"].to_numpy(), res


def cmd_pair(args):
    """通过第 2 级的方向之间算两两残差相关（任务书 §4）：逐截面在公共股票上算残差的 Spearman，再跨截面平均。"""
    h, _, _ = load(args.h)
    c = core()
    ok, did, slot = base_rows(args.start, args.end)
    store = {}
    for name in args.names:
        i_, d_, r_ = residuals_vs_h(name, h, ok, did)
        store[name] = pd.DataFrame({"idx": i_, "did": d_, "r": r_})
        print("  已算 %s" % name, file=sys.stderr)
    names = list(store)
    tab = np.full((len(names), len(names)), np.nan)
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if j < i:
                tab[i, j] = tab[j, i]
                continue
            m = store[a].merge(store[b], on="idx", suffixes=("_a", "_b"), how="inner")
            m = m.drop(columns=[c for c in ("did_b",) if c in m.columns]).rename(columns={"did_a": "did"})
            for g in [m]:
                ra = g.groupby("did")["r_a"].rank(pct=True)
                rb = g.groupby("did")["r_b"].rank(pct=True)
                da = ra - ra.groupby(g["did"]).transform("mean")
                db = rb - rb.groupby(g["did"]).transform("mean")
                c_ = (da * db).groupby(g["did"]).sum() / np.sqrt(
                    (da * da).groupby(g["did"]).sum() * (db * db).groupby(g["did"]).sum())
                tab[i, j] = float(c_[c_.notna()].mean())
    print("残差相关矩阵（对 H=%s 取残差后，逐截面 Spearman 的跨截面均值）" % args.h)
    print("            " + "".join("%9s" % n.replace("fm_", "") for n in names))
    for i, a in enumerate(names):
        print("%-11s" % a.replace("fm_", "") + "".join("%9.3f" % tab[i, j] for j in range(len(names))))
    print("\n（|相关| ≥ 0.7 视为高度重叠，只保留一个；读数写进方向总表 §4 的去重结论）")
    return tab


SUBSET_PART = {"ic_mean": 3.0, "turnover": -0.5, "ic_ir": 0.2, "ls_sharpe": 0.008}


def _series(jobname):
    d = jobdir(jobname)
    z = np.load(d / "series.npz")
    return {k: z[k] for k in z.files}


def _series_year(series, y):
    dt = np.array(series["date"]).astype("datetime64[ns]")
    yr = np.array([str(x)[:4] for x in dt])
    return {k: v[yr == y] for k, v in series.items()}


def cmd_ds(args):
    """§1.1 公榜等价分 ΔS = 3.0·ΔIC − 0.5·Δ换手 + 0.2·ΔIC_IR + 0.008·Δ多空（分年分别算）。"""
    import importlib.util
    import refs as R
    spec = importlib.util.spec_from_file_location("bscoring", str(SCORER / "build" / "scoring.py"))
    S = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(S)
    pool = R.RefPool()
    B = _series(args.h)
    for name in args.names:
        C = _series(name)
        for y in ["ALL", "2023", "2024"]:
            if y == "ALL":
                a, b = C, B
                lo, hi = pd.Period("2023-01", "M").start_time, pd.Period("2024-12", "M").end_time
            else:
                a, b = _series_year(C, y), _series_year(B, y)
                lo, hi = pd.Period("%s-01" % y, "M").start_time, pd.Period("%s-12" % y, "M").end_time
            try:
                rt = pool.metrics(lo, hi).dropna()
            except Exception:
                rt = None
            r = S.paired(None, a, b, args.h, ref_table=rt)
            if "error" in r:
                print("%-26s %-5s %s" % (name, y, r["error"]))
                continue
            d = r["diff"]
            ds = sum(SUBSET_PART[k] * d[k]["diff"] for k in SUBSET_PART if k in d)
            gate = "PASS" if ds >= 0.003 else "未过"
            parts = " ".join("%s%+.5f" % (k, d[k]["diff"]) for k in SUBSET_PART if k in d)
            print("%-26s %-5s ΔS=%+.5f [%s] | %s | n=%d corr=%.4f"
                  % (name, y, ds, gate, parts, r["n_days"], r["daily_ic_corr"]))


def ewma_along(x, did, inst, alpha):
    """对 X 在同一股票的连续截面上做 EWMA（跨日连续，α 为衰减）。返回同长度的平滑值。"""
    order = np.lexsort((did, inst))
    xs = np.asarray(x, float)[order]
    iis = inst[order]
    out = np.empty_like(xs)
    cur = np.nan
    prev = -1
    for k in range(xs.size):
        if iis[k] != prev:
            cur = np.nan
            prev = iis[k]
        v = xs[k]
        if np.isfinite(v):
            cur = v if not np.isfinite(cur) else (alpha * cur + (1 - alpha) * v)
        out[k] = cur
    res = np.empty_like(out)
    res[order] = out
    return res


def cond_quintiles(x, h, y, did, days, q=5):
    """按 X 的截面五分位分组，算每组 H 的 IC（Spearman）；返回 (各组逐截面 IC 的均值, 组内 t, 分年)。"""
    df = pd.DataFrame({"did": did, "h": h, "y": y, "x": x})
    g = df.groupby("did")
    df["g"] = (g["x"].rank(pct=True) * q).clip(upper=q - 1e-9).astype(int)
    out = {}
    for gi in range(q):
        sub = df[df["g"] == gi]
        gg = sub.groupby("did")
        rh = gg["h"].rank(pct=True)
        ry = gg["y"].rank(pct=True)
        dh = rh - rh.groupby(sub["did"]).transform("mean")
        dy = ry - ry.groupby(sub["did"]).transform("mean")
        ic = (dh * dy).groupby(sub["did"]).sum() / np.sqrt(
            (dh * dh).groupby(sub["did"]).sum() * (dy * dy).groupby(sub["did"]).sum())
        ic = ic[ic.notna()]
        out[gi] = ic
    return out


def cmd_cond(args):
    """§1.4 条件 IC：按 X 五分位分组，算 H 的 IC 与 t、分年；报最高组−最低组的差与 t。"""
    h, _, _ = load(args.h)
    c = core()
    ok, did, slot = base_rows(args.start, args.end)
    rows = ok & np.isfinite(h)
    idx0 = np.flatnonzero(rows)
    days = pd.to_datetime([pd.Timestamp(int(c.dates[int(k)])) for k in did[idx0]]).normalize()
    for name in args.names:
        x, hit, d = load(name)
        r2 = rows & np.isfinite(x)
        i2 = np.flatnonzero(r2)
        gg = cond_quintiles(x[i2], h[i2], c.y[i2], did[i2], days[np.searchsorted(idx0, i2)])
        print("== %s 条件下 H 的条件 IC（五分位，Q1=低 X，Q5=高 X）" % name)
        means = {}
        for gi in sorted(gg):
            m, t, T = nw_t(gg[gi])
            means[gi] = m
            by = {}
            print("   Q%d  IC %+.5f (t=%+.1f, n=%d)" % (gi + 1, m, t, T))
        q5, q1 = means.get(4, np.nan), means.get(0, np.nan)
        # Q5−Q1 的逐截面差与 t
        common = gg[4].index.intersection(gg[0].index)
        diff = gg[4].reindex(common) - gg[0].reindex(common)
        m, t, T = nw_t(diff)
        dids = np.array(diff.index)
        ydays = pd.to_datetime([pd.Timestamp(int(c.dates[int(k)])) for k in dids])
        yr = {}
        for y in (2023, 2024):
            sel = diff[[v.year == y for v in ydays]]
            yr[y] = nw_t(sel)[0] if len(sel) else float("nan")
        same = (yr[2023] > 0) == (yr[2024] > 0)
        print("   Q5−Q1  IC %+.5f (t=%+.1f, n=%d) 2023 %+.5f 2024 %+.5f %s"
              % (m, t, T, yr[2023], yr[2024], "PASS(t≥3且两年同向)" if (t >= 3 and same) else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ic", action="store_true")
    ap.add_argument("--res", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--slot", action="store_true")
    ap.add_argument("--pair", action="store_true")
    ap.add_argument("--ds", action="store_true")
    ap.add_argument("--cond", action="store_true")
    ap.add_argument("--ewma", default="", help="对 X 做 EWMA 的 α（逗号分隔，如 0.5,0.75,0.9）")
    ap.add_argument("--res-ewma", action="store_true", help="先对 X 逐截面剔除 H（残差）再做 EWMA（W1 的“先残差后平滑”）")
    ap.add_argument("--inter", action="store_true", help="W2 交互口径：B = H原始值·(1 + w·rank(X))")
    ap.add_argument("--h", default="fm_490_indneu25")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--ws", default="0.10,0.20,0.35,0.50")
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    if args.ic:
        cmd_ic(args)
    elif args.slot:
        cmd_slot(args)
    elif args.res:
        cmd_res(args)
    elif args.grid:
        cmd_grid(args)
    elif args.pair:
        cmd_pair(args)
    elif args.ds:
        cmd_ds(args)
    elif args.cond:
        cmd_cond(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
