# -*- coding: utf-8 -*-
"""D5 滚动学习合成（可行性）：每个月用之前 12 个月的数据拟合权重，预测当月。

B_t = 1363原始值 + Σ_j δ_j(m)·rank_j,t，m = 该交易日所属月份；δ(m) 由**严格早于该月**的
12 个月样本做岭回归（标签居中分位排名 ~ 候选居中分位排名）得到，拟合窗再往后让 5 个交易日
（防止标签窗口还没结束）。和固定权重（δ=0、S4L）比 2021–2024 各年的 IC、IC_IR、换手。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402

SPANS = [("2020-01-01", "2020-12-31", "o20"), ("2021-01-01", "2022-12-31", "o21"),
         ("2023-01-01", "2024-12-31", "fm")]
H_TASK = {"o20": "v_V1363O20_raw_wm0100", "o21": "v_V1363O21_raw_wm0100", "fm": "v_S1284z1342_raw_wm0100"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--lag-days", type=int, default=5)
    a = ap.parse_args()
    c = R.core()
    nd = c.n_dates

    H = np.full(c.n_rows, np.nan)
    for lo, hi, tag in SPANS:
        h = R.load(H_TASK[tag])[0]
        H = np.where(np.isfinite(h), h, H)
    axes = {}
    for lab, nm, nm21, nm20, al in O.CAND:
        v = np.full(c.n_rows, np.nan)
        for lo, hi, tag in SPANS:
            t = {"o20": nm20, "o21": nm21, "fm": nm}[tag]
            v = np.where(np.isfinite(v), v, O.axis(c, t, lo, hi, al))
        axes[lab] = v
        print("轴 %-8s 有效 %d" % (lab, np.isfinite(v).sum()))

    labels = list(axes)
    days = pd.to_datetime([pd.Timestamp(int(x)) for x in c.dates]).normalize()
    row_day = days[c.date_id]
    row_month = pd.Series(row_day).dt.to_period("M").to_numpy()

    def evalB(B, st, en):
        m, _ = c.evaluate(B, start=st, end=en)
        return m

    def fit_weights(mask):
        idx = np.flatnonzero(mask & np.isfinite(c.y))
        g = c.date_id[idx]
        X = np.column_stack([np.nan_to_num(axes[lab][idx]) for lab in labels])
        ryc = O.rc_of(c.y[idx], g, nd)
        good = np.isfinite(ryc)
        idx, g, X, ryc = idx[good], g[good], X[good], ryc[good]
        cnt = np.bincount(g, minlength=nd).astype(float)
        cnt = np.where(cnt > 0, cnt, 1.0)
        for j in range(X.shape[1]):
            X[:, j] -= np.bincount(g, X[:, j], nd)[g] / cnt[g]
        ryc = ryc - np.bincount(g, ryc, nd)[g] / cnt[g]
        return np.linalg.solve(X.T @ X + a.ridge * np.eye(X.shape[1]), X.T @ ryc), len(idx)

    months = pd.period_range("2021-01", "2024-12", freq="M")
    B = H.copy()
    wlog = {}
    for m in months:
        hist = (row_month >= m - 12) & (row_month < m)
        # 让 5 个交易日：去掉拟合窗最后 lag-days 个交易日
        hd = np.unique(row_day[hist])
        if len(hd) <= a.lag_days + 30:
            continue
        cut = hd[-a.lag_days] if a.lag_days else hd[-1] + pd.Timedelta(days=1)
        hist = hist & (row_day < cut)
        w, n = fit_weights(hist)
        sel = (row_month == m) & np.isfinite(H)
        for j, lab in enumerate(labels):
            if w[j]:
                B[sel] = B[sel] + w[j] * np.nan_to_num(axes[lab][sel])
        wlog[str(m)] = w
    B[pd.Series(row_day).dt.strftime("%H%M").to_numpy() == "0930"]=0.0

    print("\n== 逐年 ==")
    print("%-10s %-28s %-28s %-28s" % ("年份", "滚动学习", "固定 1363", "固定 S4L"))
    greedy = {"s155": 0.25, "s139": 0.10, "k1381": 0.02, "k1385": 0.02, "k1406": 0.05}
    Hs = H
    for y in (2021, 2022, 2023, 2024):
        st, en = "%d-01-01" % y, "%d-12-31" % y
        mB = evalB(B, st, en)
        mH = evalB(Hs, st, en)
        Bg = Hs.copy()
        for lab, wv in greedy.items():
            Bg = Bg + wv * np.nan_to_num(axes[lab])
        mG = evalB(Bg, st, en)
        print("%-10d IC %.5f IR %.3f 换手 %.3f | IC %.5f IR %.3f 换手 %.3f | IC %.5f IR %.3f 换手 %.3f"
              % (y, mB["ic_mean"], mB["ic_ir"], mB["turnover"],
                 mH["ic_mean"], mH["ic_ir"], mH["turnover"],
                 mG["ic_mean"], mG["ic_ir"], mG["turnover"]))
    print("\n权重的年度均值（前 6 个绝对值最大的）：")
    wt = np.array([v for v in wlog.values()])
    order = np.argsort(-np.abs(wt).mean(axis=0))
    for j in order[:6]:
        print("   %-8s 均值 %+.4f  标准差 %.4f" % (labels[j], wt[:, j].mean(), wt[:, j].std()))


if __name__ == "__main__":
    main()
