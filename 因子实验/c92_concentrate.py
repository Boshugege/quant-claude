# -*- coding: utf-8 -*-
"""集中优化（10-20 晚，personal agent.slice 上跑）：1610 + 纯方向主题的分量，带换手惩罚的线性定权。
分量（各自逐截面居中；快信号先平滑）：
  B1610   1610 原始合成值（161 头部、七根、三小根、收盘段季节性、收盘位置）
  主题：波动标准化特质段反转、特质放量／跳跃、特质主动买压、特质盘口订单流（平滑 0.6）、特质 VWAP 偏离（0.3）、
        委托结构（深档笔数 + 一档规模比，0.3）、特质日内反转、订单流常态、价差常态、微观价格常态、笔数失衡常态
定权：min ‖y − Xw‖² + λ·wᵀDw，D = Σ(同一股票同一交易日相邻截面的分量差的外积)（换手惩罚，闭式解）。
y = 秩(标签) + 0.5·秩(之后 2 个截面标签和)；2020–22 定权，λ 按 tr(XᵀX)/tr(D) 标定后取 {0, 0.5, 1, 2, 4, 8}。
与 1610 比五窗口五项（★ = 不差），打印权重；导出 2024 因子值送评分机估分。"""
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, CLS, CP, build, MET, OUT
from g_go import inst_codes
c = G.c; z = G.z930; did = c.date_id; slot = c.slot_of_date[did]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask=None):
    x = np.where((np.isfinite(x) & (x != 0)) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
sm = lambda x, a: H.ewma_col(x, c.inst, a)                                                   # noqa: E731
cnt = np.maximum(np.bincount(did, minlength=c.n_dates), 1)
cen = lambda x: x - (np.bincount(did, x, c.n_dates) / cnt)[did]                              # noqa: E731
yrs = pd.to_datetime(c.key_date).year.to_numpy()
B = {ix: np.nan_to_num(np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP), nan=0.0) for ix in (0, 1, 2)}   # 缺值记 0（否则标准差与回归全是 NaN）
REF = {wn: G.ev(B[ix], wn) for wn, ix in W.items()}
DR, PS = raw("D_R"), raw("P_SEG"); mP = (DR != 0) & (PS != 0)
ID = np.where(mP, DR - PS, np.nan)
order = np.lexsort((pd.to_datetime(c.key_date).values.astype(np.int64), c.inst))
sd = pd.Series(np.where(z, np.nan, ID)[order]).groupby(c.inst[order]).transform(lambda q: q.rolling(140, min_periods=40).std().shift(1)).to_numpy()
sig = np.empty_like(sd); sig[order] = sd
TH = {"波动标准化特质段反转": sm(-rk(ID / sig), 0.6), "特质放量跳跃": sm(-rk(raw("D_HV") + raw("D_JUMP") - PS, mP), 0.6),
      "特质主动买压": sm(-rk(raw("PD_MS_OFI")), 0.6), "特质盘口订单流": sm(-rk(raw("PD_CF_OFI3")), 0.6),
      "特质VWAP偏离": sm(-rk(raw("PD_AN_VWAP")), 0.3), "委托结构": sm(np.mean([rk(raw("QL_NIMB23")), -rk(raw("OQ_SIZE"))], axis=0), 0.3),
      "特质日内反转": rk(raw("P_CUM")), "订单流常态": rk(raw("CF_OFI3_20")), "价差常态": rk(raw("AN_SPRD_20")),
      "微观价格常态": rk(raw("AN_MICRO_20")), "笔数失衡常态": rk(raw("QL_NIMBA_20"))}
names = ["B1610"] + list(TH)
s1610 = np.nanstd(B[0][~z])
def Xof(ix):
    return np.column_stack([cen(B[ix] / s1610)] + [cen(TH[k]) for k in TH])
rowix = np.where(yrs >= 2023, 0, np.where(yrs >= 2021, 1, 2))
fit = (yrs >= 2020) & (yrs <= 2022) & ~z
Xf = np.where((rowix == 1)[:, None], Xof(1), Xof(2))[fit]
y0 = np.nan_to_num(O.rc_of(c.y, did, c.n_dates), nan=0.0)
d = pd.DataFrame({"inst": c.inst, "day": pd.to_datetime(c.key_date).normalize(), "y": c.y}).iloc[order]
g = d.groupby(["inst", "day"], sort=False)["y"]
nx = (g.shift(-1).fillna(0.0) + g.shift(-2).fillna(0.0)).where(g.shift(-1).notna())
ysum = np.full(c.n_rows, np.nan); ysum[order] = nx.to_numpy()
y = cen(y0 + 0.5 * np.nan_to_num(O.rc_of(ysum, did, c.n_dates), nan=0.0))[fit]
fi = np.flatnonzero(fit); o2 = fi[np.lexsort((pd.to_datetime(c.key_date).values.astype(np.int64)[fi], c.inst[fi]))]
pos = np.empty(c.n_rows, np.int64); pos[fi] = np.arange(len(fi))
same = (c.inst[o2][1:] == c.inst[o2][:-1]) & (pd.to_datetime(c.key_date[o2][1:]).normalize() == pd.to_datetime(c.key_date[o2][:-1]).normalize())
dX = Xf[pos[o2[1:]]][same] - Xf[pos[o2[:-1]]][same]
XtX, Xty, D = Xf.T @ Xf, Xf.T @ y, dX.T @ dX
scale = np.trace(XtX) / np.trace(D)
kd = pd.to_datetime(c.key_date); m24 = (kd.year == 2024); ins = inst_codes(c)
for lam in tuple(float(v) for v in __import__("os").environ.get("LAMS", "0,0.5,1,2,4,8").split(",")):
    w = np.linalg.solve(XtX + lam * scale * D + 1e-6 * np.trace(XtX) * np.eye(len(names)), Xty)
    w = w / abs(w[0]) if w[0] != 0 else w
    print("λ=%.1f 权重（1610 = 1）：" % lam + "，".join("%s %+.3f" % (k, v) for k, v in zip(names, w)), flush=True)
    line, st, comp = "    相对 1610", 0, {}
    for wn, ix in W.items():
        comp[ix] = comp.get(ix) if ix in comp else np.where(z, 0.0, Xof(ix) @ w)
        r = G.ev(comp[ix], wn); dd = {m: r[m] - REF[wn][m] for m in MET}
        ok = dd["ic_mean"] >= -0.00005 and dd["ic_ir"] >= 0 and dd["ls_sharpe"] >= 0 and dd["stress"] >= 0 and dd["turnover"] <= 0
        st += ok
        line += " | %s %+.4f/%+.3f/%+.2f/%+.3f/%+.4f%s" % (wn, dd["ic_mean"], dd["ic_ir"], dd["ls_sharpe"], dd["stress"], dd["turnover"], "★" if ok else "")
    print(line + "  %d/5" % st, flush=True)
    o = pd.DataFrame({"date": kd[m24], "instrument": ins[np.flatnonzero(m24)], "factor": comp[0][m24]}).drop_duplicates(["date", "instrument"], keep="last")
    o.to_parquet(OUT / ("val_K_lam%03d_2024.parquet" % round(lam * 100)), index=False)
