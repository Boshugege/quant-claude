# -*- coding: utf-8 -*-
"""训练中 [val] 的指标，口径与平台 jyc_eval 一致（与 工具/本地评分/localeval.evaluate 相同，见 reports/12）：
- 因子处理：逐截面 inf 处理 → 标准化 → 对当日 10 个官方风格 + 申万一级行业哑变量（含截距）OLS 取残差（style 传入这些列）；没有官方暴露的股票不参与评估。
- IC：残差因子与标签的截面 Spearman 秩相关；IC_IR = 均值 / 标准差。
- 多空夏普：前 10% 减后 10% 的等权标签收益，按每年 242 × 8 个截面年化。
- 压力：按截面标签离散度中位数分高低两组，取较弱一组的 IC_IR。
- 换手：多空两腿（各 10%）在同一交易日内相邻截面之间的成分替换比例（跨日不计），两腿平均。
style=None 时跳过风格剔除（只作对照）。
"""
import math

import numpy as np
import pandas as pd


def style_residual(keys, pred, style):
    v = pd.Series(np.asarray(pred, np.float64)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    B = np.asarray(style, np.float64)
    has = np.isfinite(B).all(1)
    out = np.full(len(v), np.nan)
    for _, ii in pd.Series(np.arange(len(v))).groupby(keys["date"].values):
        ii = ii.to_numpy()
        ii = ii[has[ii]]
        x = v[ii]
        sd = x.std() if len(ii) else 0.0
        if len(ii) < 50 or not sd > 0:
            continue
        z = (x - x.mean()) / sd
        X = np.column_stack([np.ones(len(ii)), B[ii]])
        coef, *_ = np.linalg.lstsq(X, z, rcond=None)
        out[ii] = z - X @ coef
    return out


def evaluate(keys, pred, y, style=None, min_stocks=100):
    """keys[date, instrument] + 预测 + 标签 -> IC 均值、IC_IR、多空夏普、压力、换手率。"""
    f = style_residual(keys, pred, style) if style is not None else \
        pd.Series(np.asarray(pred, np.float64)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    d = keys[["date", "instrument"]].assign(f=f, y=y).dropna(subset=["y", "f"])
    ics, ls, turn, ysd, prev = [], [], [], [], None
    for t, g in d.groupby("date", sort=True):
        if len(g) < min_stocks:
            continue
        ics.append(g["f"].rank().corr(g["y"].rank()))
        ysd.append(g["y"].std())
        n = max(int(len(g) * 0.1), 1)
        o = g.sort_values("f")
        top, bot = set(o["instrument"].iloc[-n:]), set(o["instrument"].iloc[:n])
        ls.append(o["y"].iloc[-n:].mean() - o["y"].iloc[:n].mean())
        day = pd.Timestamp(t).normalize()
        if prev is not None and prev[2] == day:
            turn.append((1 - len(top & prev[0]) / n + 1 - len(bot & prev[1]) / n) / 2)
        prev = (top, bot, day)
    ics, ls, ysd = np.array(ics), np.array(ls), np.array(ysd)

    def ir(x):
        return float(np.nanmean(x) / np.nanstd(x)) if len(x) > 1 and np.nanstd(x) > 0 else float("nan")
    hi = ysd > np.median(ysd) if len(ysd) else np.array([], bool)
    return {
        "sections": int(len(ics)),
        "ic": float(np.nanmean(ics)),
        "icir": ir(ics),
        "ls_sharpe": float(ls.mean() / ls.std() * math.sqrt(242 * 8)),
        "stress": min(ir(ics[hi]), ir(ics[~hi])) if len(ics) else float("nan"),
        "turnover": float(np.mean(turn)) if turn else float("nan"),
    }
