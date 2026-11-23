# -*- coding: utf-8 -*-
"""主题"流动性冲击回吐"第四轮（10-20）：主题内部定权（同 1610 的做法：线性、分量与权重有记录，符号须与机制一致）。
分量（截面秩，快信号各自平滑 α=0.6；符号按机制）：
  特质段反转、特质放量／跳跃反转、特质主动买压、特质盘口订单流、特质 VWAP 偏离（α=0.3）、特质区间位置（α=0.3）、
  特质日内反转、订单流常态、委托结构（深档笔数 + 一档规模比，α=0.3）、价差常态、微观价格常态、笔数失衡常态
目标 = 秩(标签) + w·秩(同日之后 2 个截面标签之和)（w 大 = 偏向持续的信号 → 换手低），w ∈ {0, 0.5, 1.0}；
2020–22 上 OLS（小岭），2023／2024 检验；导出 2024 因子值送评分机。"""
import numpy as np
import pandas as pd
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, MET, OUT
from g_go import inst_codes
c = G.c; z = G.z930; did = c.date_id
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask=None):
    x = np.where((x != 0) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
sm = lambda x, a: H.ewma_col(x, c.inst, a)                                   # noqa: E731
mD = raw("D_R") != 0; PS = raw("P_SEG"); mP = mD & (PS != 0)
FE = {
    "特质段反转": sm(-rk(raw("D_R") - PS, mP), 0.6), "特质放量跳跃": sm(-rk(raw("D_HV") + raw("D_JUMP") - PS, mP), 0.6),
    "特质主动买压": sm(-rk(raw("PD_MS_OFI")), 0.6), "特质盘口订单流": sm(-rk(raw("PD_CF_OFI3")), 0.6),
    "特质VWAP偏离": sm(-rk(raw("PD_AN_VWAP")), 0.3), "特质区间位置": sm(-rk(raw("PD_AN_RPOS")), 0.3),
    "特质日内反转": rk(raw("P_CUM")), "订单流常态": rk(raw("CF_OFI3_20")),
    "委托结构": sm(np.mean([rk(raw("QL_NIMB23")), -rk(raw("OQ_SIZE"))], axis=0), 0.3),
    "价差常态": rk(raw("AN_SPRD_20")), "微观价格常态": rk(raw("AN_MICRO_20")), "笔数失衡常态": rk(raw("QL_NIMBA_20")),
}
names = list(FE)
yrs = pd.to_datetime(c.key_date).year.to_numpy()
fit = (yrs >= 2020) & (yrs <= 2022) & ~z
def secmean(x):
    return (np.bincount(did, x, c.n_dates) / np.maximum(np.bincount(did, minlength=c.n_dates), 1))[did]
y0 = np.nan_to_num(O.rc_of(c.y, did, c.n_dates), nan=0.0)
d = pd.DataFrame({"inst": c.inst, "sec": did, "day": pd.to_datetime(c.key_date).normalize(), "y": c.y})
d = d.sort_values(["inst", "sec"])
g = d.groupby(["inst", "day"], sort=False)["y"]
nx = g.shift(-1).fillna(0.0) + g.shift(-2).fillna(0.0)
ysum = np.full(c.n_rows, np.nan); ysum[d.index.to_numpy()] = nx.where(g.shift(-1).notna(), np.nan).to_numpy()
ya = np.nan_to_num(O.rc_of(ysum, did, c.n_dates), nan=0.0)
X = np.column_stack([FE[k] - secmean(FE[k]) for k in names])
kd = pd.to_datetime(c.key_date); m24 = (kd.year == 2024); ins = inst_codes(c)
for wa in (0.0, 0.5, 1.0):
    y = y0 + wa * ya
    y = y - secmean(y)
    Xf = X[fit]
    w = np.linalg.solve(Xf.T @ Xf + 1e3 * np.eye(len(names)), Xf.T @ y[fit])
    w = w / np.abs(w).sum()
    print("目标多期权重 %.1f 的分量权重：" % wa + "，".join("%s %+.3f" % (k, v) for k, v in zip(names, w)), flush=True)
    x = np.where(z, 0.0, X @ w)
    r = {wn: G.ev(x, wn) for wn in ("2023", "2024", "2021", "2022", "2020")}
    print("    " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][k] for k in MET]) for wn in r), flush=True)
    for sa in (0.0, 0.3):
        xs = np.where(z, 0.0, sm(x, sa)) if sa > 0 else x
        tag = "H_fit_a%02d_s%02d" % (wa * 10, sa * 10)
        if sa > 0:
            r = {wn: G.ev(xs, wn) for wn in ("2023", "2024")}
            print("    再平滑 %.1f：" % sa + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][k] for k in MET]) for wn in r), flush=True)
        o = pd.DataFrame({"date": kd[m24], "instrument": ins[np.flatnonzero(m24)], "factor": xs[m24]}).drop_duplicates(["date", "instrument"], keep="last")
        o.to_parquet(OUT / ("val_%s_2024.parquet" % tag), index=False)
        print("    导出", tag, flush=True)
