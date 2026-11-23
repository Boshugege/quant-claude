# -*- coding: utf-8 -*-
"""方向 B（10-20）：条件反转——把树在反转头部上学到的非线性，换成少数几个有含义的显式交互项。
头部 h ∈ {161 头部（同伴比较的价格锚点反转）、490 头部（多尺度反转阶梯）}，均为截面居中秩。
1）形状：按 h 分十档，目标秩（及扣掉 1610 后的残差）的档均值，拟合期 2020–22 与样本外 2023–24 各一份。
2）单个交互项：1610 + w·项，w 由 2020–22 的二元 OLS 定（目标 = 标签秩），与 1610 比五窗口五项。
   项：h·|h|（尾部弯曲）、h × 各时段、h × 放量占比、h × 特质占比、h × 量能冲击（VSH20）、h × 当段市场涨跌。
3）全部交互项一起 OLS（只报权重与样本外）。"""
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
import c52_mech_group as M
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; did = c.date_id; slot = c.slot_of_date[did]
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
SLOTS = (1000, 1030, 1100, 1130, 1330, 1400, 1430)
yrs = pd.to_datetime(c.key_date).year.to_numpy()
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
rk = lambda x: np.where(z, 0.0, np.nan_to_num(O.rc_of(np.where(np.isfinite(x), x, np.nan), did, c.n_dates), nan=0.0))
def secmean(x):
    return np.bincount(did, x, c.n_dates)[did] / np.maximum(np.bincount(did, minlength=c.n_dates), 1)[did]
cen = lambda x: x - secmean(x)
y = cen(np.nan_to_num(O.rc_of(c.y, did, c.n_dates), nan=0.0))
DR, DH, DL, DJ, PS = raw("D_R"), raw("D_HV"), raw("D_LV"), raw("D_JUMP"), raw("P_SEG")
ok = (DR != 0)
den = np.abs(DH) + np.abs(DL) + np.abs(DJ)
HVS = rk(np.where(ok & (den > 0), np.abs(DH) / np.where(den > 0, den, 1.0), np.nan))      # 放量占比
ID = np.abs(DR - PS); IDS = rk(np.where(ok & (PS != 0), ID / (ID + np.abs(PS) + 1e-12), np.nan))   # 特质占比
VS = rk(np.where(raw("VSH20") != 0, raw("VSH20"), np.nan))                                 # 量能冲击
MK = (np.bincount(did, np.where(ok, DR, 0.0), c.n_dates) / np.maximum(np.bincount(did, ok.astype(float), c.n_dates), 1))[did]
MK = np.where(z, 0.0, MK / (MK[~z].std() + 1e-12))                                         # 当段市场等权涨跌（标准化，截面内为常数）
U.DEAL = {"Y_sd5", "X_n5d"}
base = {}; heads = {}
for ix in (0, 1, 2):
    base[ix] = np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP)
    a = M.groups(ix)["锚点反转"]
    heads[ix] = {"161": a[0], "490": a[1]}
ref = {wn: G.ev(base[ix], wn) for wn, ix in W.items()}
def terms(ix):
    T = {}
    for hn, h in heads[ix].items():
        T[hn + "·|h|"] = h * np.abs(h)
        for s in SLOTS:
            T[hn + "×%d" % s] = h * (slot == s)
        T[hn + "×放量占比"] = h * HVS
        T[hn + "×特质占比"] = h * IDS
        T[hn + "×量能冲击"] = h * VS
        T[hn + "×市场涨跌"] = h * MK
    return T
TT = {ix: terms(ix) for ix in (0, 1, 2)}
rowix = np.where(yrs >= 2023, 0, np.where(yrs >= 2021, 1, 2))
pick = lambda d: np.select([rowix == 0, rowix == 1, rowix == 2], [d[0], d[1], d[2]])
fit = (yrs >= 2020) & (yrs <= 2022) & ~z
oos = (yrs >= 2023) & ~z
# 1）形状
for hn in ("161", "490"):
    h = pick({ix: heads[ix][hn] for ix in (0, 1, 2)})
    b = cen(pick({ix: rk(base[ix]) for ix in (0, 1, 2)}))
    beta = (b[fit] @ y[fit]) / (b[fit] @ b[fit]); res = y - beta * b
    q = np.clip(((h + 0.5) * 10).astype(int), 0, 9)
    for tag, m in (("拟合 2020–22", fit), ("样本外 2023–24", oos)):
        mu = [y[m & (q == k)].mean() for k in range(10)]; mr = [res[m & (q == k)].mean() for k in range(10)]
        print("%s 头部十档（%s）目标均值：%s" % (hn, tag, " ".join("%+.4f" % v for v in mu)), flush=True)
        print("%s 头部十档（%s）扣 1610 残差：%s" % (hn, tag, " ".join("%+.4f" % v for v in mr)), flush=True)
print("%-22s " % "1610" + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
def ols(cols):
    X = np.column_stack(cols)[fit]
    return np.linalg.solve(X.T @ X + 1e-9 * np.eye(X.shape[1]), X.T @ y[fit])
B = {ix: cen(rk(base[ix])) for ix in (0, 1, 2)}
Bp = pick(B)
def show(nm, comp):
    st, line = 0, "%-22s " % nm
    for wn, ix in W.items():
        r = G.ev(np.where(z, 0.0, comp[ix]), wn); d = {m: r[m] - ref[wn][m] for m in MET}
        st += M.ok(d)
        line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if M.ok(d) else "")
    print(line + "  %d/5" % st, flush=True)
# 2）单个交互项
for k in TT[0]:
    tk = cen(pick({ix: TT[ix][k] for ix in (0, 1, 2)}))
    w = ols([Bp, tk])
    show("+%s（%+.3f）" % (k, w[1] / w[0]), {ix: B[ix] + (w[1] / w[0]) * cen(TT[ix][k]) for ix in (0, 1, 2)})
# 3）分组一起
GROUPS = {"尾部弯曲": [k for k in TT[0] if "|h|" in k], "时段": [k for k in TT[0] if "×1" in k],
          "放量占比": [k for k in TT[0] if "放量" in k], "特质占比": [k for k in TT[0] if "特质" in k],
          "量能冲击": [k for k in TT[0] if "量能" in k], "市场涨跌": [k for k in TT[0] if "市场" in k]}
GROUPS["全部"] = list(TT[0])
for gn, ks in GROUPS.items():
    cols = [Bp] + [cen(pick({ix: TT[ix][k] for ix in (0, 1, 2)})) for k in ks]
    w = ols(cols); w = w / w[0]
    if gn == "全部":
        print("全部交互项权重（相对 1610 = 1）：" + "，".join("%s %+.3f" % (k, v) for k, v in zip(ks, w[1:])), flush=True)
    show("组 " + gn, {ix: B[ix] + sum(wk * cen(TT[ix][k]) for wk, k in zip(w[1:], ks)) for ix in (0, 1, 2)})
