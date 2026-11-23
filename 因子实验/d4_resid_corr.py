# -*- coding: utf-8 -*-
"""D4 第三个头部：候选头部对底座的残差相关矩阵 + 残差 IC（对底座的新增信息）。

底座默认 `v_SB155_raw_wp0500`（1363 + s155×0.5）。逐截面把候选的居中分位排名对底座的居中分位
排名做 OLS 取残差，再算残差之间的逐截面相关（对截面取均值）。候选覆盖率不同，**两两**在共同
行上算，不做整表求交。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import engine as E  # noqa: E402

CAND = ["fm_490_indneu25", "fm_1083", "fm_233_win_upsh30", "fm_237_upsh30n2",
        "fm_390_ewma_slotw", "fm_1176", "fm_1187", "fm_1391",
        "fm_s137", "fm_s139", "fm_s160", "fm_s161"]


SL930 = None      # 09:30 截面（rank 轴按合成脚本口径置 0，避免多算一个截面）


def rc(vals, gid, nd, sl=None):
    m = np.isfinite(vals)
    out = np.full(len(vals), np.nan)
    if m.any():
        rp, _ = E._group_rank_pct(gid[m], vals[m], nd)
        out[m] = rp - 0.5
    if sl is not None:
        out[sl] = 0.0
    return out


def resid_on(base, f, gid, nd):
    m = np.isfinite(base) & np.isfinite(f)
    idx = np.flatnonzero(m)
    gg = gid[idx]
    sl = None if SL930 is None else SL930[idx]
    rh = rc(np.asarray(base[idx], float), gg, nd, sl)
    rx = rc(np.asarray(f[idx], float), gg, nd, sl)
    ok = np.isfinite(rh) & np.isfinite(rx)
    idx, gg, rh, rx = idx[ok], gg[ok], rh[ok], rx[ok]
    s_hh = np.bincount(gg, rh * rh, nd)
    beta = np.bincount(gg, rx * rh, nd) / np.where(s_hh > 0, s_hh, np.nan)
    return idx, gg, rx - beta[gg] * rh


def mean_corr_fast(gid_all, n_rows, nd, ei, ej, min_n=100):
    """逐截面皮尔逊相关再对截面取均值（与 mean_corr 等价，但走 bincount 向量化）。"""
    ai = np.full(n_rows, np.nan)
    aj = np.full(n_rows, np.nan)
    ai[ei[0]] = ei[1]
    aj[ej[0]] = ej[1]
    m = np.isfinite(ai) & np.isfinite(aj)
    if not m.any():
        return float("nan")
    g = gid_all[m]
    x = ai[m]
    y = aj[m]
    pi = np.bincount(g, weights=x * y, minlength=nd)
    si = np.bincount(g, weights=x * x, minlength=nd)
    sj = np.bincount(g, weights=y * y, minlength=nd)
    nn = np.bincount(g, minlength=nd)
    den = np.sqrt(si * sj)
    ok = (den > 0) & (nn >= min_n)
    return float((pi[ok] / den[ok]).mean()) if ok.any() else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="v_SB155_raw_wp0500")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default="2024-12-31")
    a = ap.parse_args()
    c = R.core()
    nd = c.n_dates
    globals()["SL930"] = (c.slot_of_date[c.date_id] == 930)
    _ok, did, _slot = R.base_rows(a.start, a.end)
    base, _h, _d = R.load(a.base)
    store = {}
    for nm in CAND:
        try:
            f, _h2, _d2 = R.load(nm)
        except SystemExit:
            print("跳过（找不到）", nm)
            continue
        idx, gg, res = resid_on(base, f, did, nd)
        store[nm] = (idx, gg, res)
        print("覆盖 %-20s 行 %d" % (nm, len(idx)))
    names = list(store)
    K = len(names)
    gid_all = c.date_id
    n_rows = c.n_rows
    C = np.full((K, K), np.nan)
    for i in range(K):
        C[i, i] = 1.0
        for j in range(i + 1, K):
            _, _gi, ri = store[names[i]]
            _, _gj, rj = store[names[j]]
            C[i, j] = C[j, i] = mean_corr_fast(
                gid_all, n_rows, nd, (store[names[i]][0], ri), (store[names[j]][0], rj))
    print("\n底座 %s（%s~%s）对底座取残差后的逐截面相关（均值）：" % (a.base, a.start, a.end))
    print("%-20s%s" % ("", "".join("%9s" % n.replace("fm_", "")[:8] for n in names)))
    for i, n in enumerate(names):
        print("%-20s%s" % (n, "".join("%9.3f" % C[i, j] for j in range(K))))
    print("\n候选对底座的残差 IC（平台口径，风格+行业剔除后）：")
    for nm in names:
        idx, gg, res = store[nm]
        f = np.full(c.n_rows, np.nan)
        f[idx] = res
        m, _s = c.evaluate(f, start=a.start, end=a.end)
        print("  %-20s IC %+.5f IR %+.4f 压力 %+.4f" % (nm, m["ic_mean"], m["ic_ir"], m["stress"]))


if __name__ == "__main__":
    main()
