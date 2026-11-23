# -*- coding: utf-8 -*-
"""D2／D3／D5 共用的离线评估器：用评分机 engine 的**精确口径**算五项指标。

B = 1363原始值 + Σ_j δ_j · rank(c_j)   （rank 为逐截面分位中心化，与 values 合成一致）
evaluate() 直接调 engine.Core.evaluate（风格+行业剔除、压力=标签离散度高低两组的 IC_IR 取小、换手=两腿相邻截面替换比例）。
"""
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402

import json as _json
import re as _re
_JOBS = [pathlib.Path("/home/user/jyc_scorer/jobs"), pathlib.Path("/srv/media/jyc_scorer/jobs_archive")]


def jid_name(jid):
    """任务号 → 任务名（root_eval.load 按名字查）。"""
    for b in _JOBS:
        sp = b / jid / "spec.json"
        if sp.exists():
            return _json.load(open(sp, encoding="utf-8"))["name"]
    raise SystemExit("找不到任务号 %s" % jid)

CORE = None
PA = None          # 每行的 (date_id, instrument) 分组键


def core():
    global CORE
    if CORE is None:
        CORE = R.core()
    return CORE


def aligned(name):
    if _re.match(r"^\d{4}-\d{6}-[0-9a-z]{4}$", name):
        name = jid_name(name)
    f, hit, d = R.load(name)
    return f


def rank_axis(f):
    """逐截面分位排名居中（平均秩百分位 − 0.5，与 values 合成一致）；缺失记 0。"""
    import engine as _E
    c = core()
    sel = np.flatnonzero(np.isfinite(f))
    out = np.zeros(c.n_rows)
    if len(sel):
        rp, _n = _E._group_rank_pct(c.date_id[sel], np.asarray(f[sel], float), c.n_dates)
        out[sel] = rp - 0.5
    out[c.slot_of_date[c.date_id] == 930] = 0.0     # 与合成脚本一致，见 d2_optimize.axis
    return out


def evaluate(B, start, end):
    m, _s = core().evaluate(B, start=start, end=end)
    return m


def show(tag, m):
    return ("%s IC %+.5f IR %+.4f LS %+.2f 压力 %+.4f 换手 %.4f (n=%d)"
            % (tag, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"], m["n_xsec"]))


if __name__ == "__main__":
    import time
    t0 = time.time()
    f = aligned("1015-203626-f5f0")
    print("load 1363 rows=%d %.0fs" % (np.isfinite(f).sum(), time.time() - t0))
    for st, en in (("2020-01-01", "2022-12-31"), ("2023-01-01", "2024-12-31")):
        t1 = time.time()
        m = evaluate(f, st, en)
        print("%s~%s  %s  [%.1fs]" % (st, en, show("1363", m), time.time() - t1))


# ---------------------------------------------------------------- 提速：预先做残差，逐次只重算截面排名
def resid_axis(name, start, end):
    """风格+行业剔除后的残差轴（与评测机 series() 的 res 完全一致）。"""
    import engine as _E  # noqa: F401
    c = core()
    f = aligned(name) if isinstance(name, str) else name
    dm = c.date_range_mask(start, end)
    return c.residual(f, dm)


def series_from_res(res, start, end):
    """复制 engine.Core.series（neutralize 之后的部分），输入已经是残差。"""
    c = core()
    dm = c.date_range_mask(start, end)
    y = c.y
    sel = np.flatnonzero(np.isfinite(res) & np.isfinite(y))
    g = c.date_id[sel]
    D = c.n_dates
    rf, n = _E._group_rank_pct(g, res[sel], D)
    ry, _ = _E._group_rank_pct(g, y[sel], D)
    nz = np.maximum(n, 1)
    xf = rf - (np.bincount(g, rf, D) / nz)[g]
    yf = ry - (np.bincount(g, ry, D) / nz)[g]
    xy, xx, yy = np.bincount(g, xf * yf, D), np.bincount(g, xf * xf, D), np.bincount(g, yf * yf, D)
    with np.errstate(invalid="ignore", divide="ignore"):
        ic = xy / np.sqrt(xx * yy)
    ok = (n >= 100) & np.isfinite(ic)
    ysel = y[sel]
    top, bot = rf > 0.9, rf <= 0.1
    with np.errstate(invalid="ignore", divide="ignore"):
        ls = (np.bincount(g, ysel * top, D) / np.bincount(g, top, D)
              - np.bincount(g, ysel * bot, D) / np.bincount(g, bot, D))
        ym = np.bincount(g, ysel, D) / nz
        ysd = np.sqrt(np.bincount(g, (ysel - ym[g]) ** 2, D) / (n - 1))
    kept = np.flatnonzero(ok)
    tl = c._turnover(kept, g, top, sel)
    ts = c._turnover(kept, g, bot, sel)
    return {"g": kept, "date": c.dates[kept], "day": c.day_of_date[kept], "slot": c.slot_of_date[kept],
            "ic": ic[kept], "ls": ls[kept], "ysd": ysd[kept], "n": n[kept], "tl": tl, "ts": ts}


import engine as _E  # noqa: E402


def eval_mix(base_res, pairs, start, end):
    """base_res + Σ w·R_j 的五项指标。pairs = [(w, R_j), ...]"""
    res = base_res
    for w, Rj in pairs:
        if w:
            res = res + w * Rj
    s = series_from_res(res, start, end)
    return core().summarize(s), s


if True and __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "--bench":
    import time
    st, en = "2020-01-01", "2022-12-31"
    t0 = time.time()
    RH = resid_axis("1015-203626-f5f0", st, en)
    print("resid H %.1fs" % (time.time() - t0))
    t0 = time.time()
    m, _ = eval_mix(RH, [], st, en)
    print("eval %.1fs %s" % (time.time() - t0, show("H", m)))
    t0 = time.time()
    m2 = evaluate(aligned("1015-203626-f5f0"), st, en)
    print("full %.1fs %s" % (time.time() - t0, show("Hfull", m2)))
