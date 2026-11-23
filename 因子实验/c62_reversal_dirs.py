# -*- coding: utf-8 -*-
"""方向 A（分解式反转）与 C（特质反转阶梯）第一版评估（10-20）。都是独立方向，与 1610 比五窗口五项，不做叠加。
原料：m12（段收益拆成跳跃／放量／缩量三块，及当日累计）、m11（P_SEG、P_CUM、P_DAY）、m11b（P_GAP、P_5D、P_20D、O_5D）、m9（I_INTRA、I_GAP）。
1）各分量单独：反转方向的 IC 等五项（检验"流动性冲击反转、信息延续"）。
2）合成：先验等权版（符号来自机制），以及在 2020–22 用 OLS 定权的版本（截面秩为特征、标签秩为目标；只报权重和符号，不调参）。"""
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; did = c.date_id; slot = c.slot_of_date[did]
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask):
    x = np.where(mask & ~z, x, np.nan)
    return np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0)
mD = raw("D_R") != 0; mC = raw("C_R") != 0
F = {}
for nm in ("D_R", "D_JUMP", "D_HV", "D_LV"):
    F[nm] = rk(raw(nm), mD)
for nm in ("C_R", "C_JUMP", "C_HV", "C_LV"):
    F[nm] = rk(raw(nm), mC)
for nm in ("P_SEG", "P_CUM", "P_DAY", "P_GAP", "P_5D", "P_20D", "O_5D", "I_INTRA", "I_GAP"):
    x = raw(nm); F[nm] = rk(x, x != 0)
F["D_IDIO"] = rk(raw("D_R") - raw("P_SEG"), mD & (raw("P_SEG") != 0))           # 段收益扣同伴
F["D_NOJ"] = rk(raw("D_R") - raw("D_JUMP"), mD)                                 # 段收益去掉跳跃
F["C_NOJ"] = rk(raw("C_R") - raw("C_JUMP"), mC)
print("分量单独（取值越大 = 原始值越大；IC 为正 = 延续，为负 = 反转）", flush=True)
for nm, x in F.items():
    r = {wn: G.ev(x, wn) for wn in W}
    print("%-8s " % nm + " | ".join("%s %+.4f/%+.3f/%+.1f/%+.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
U.DEAL = {"Y_sd5", "X_n5d"}
ref = {wn: G.ev(np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP), wn) for wn, ix in W.items()}
print("%-30s " % "1610" + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
yr = np.nan_to_num(O.rc_of(c.y, did, c.n_dates), nan=0.0)
yrs = pd.to_datetime(c.key_date).year.to_numpy()
fit = (yrs >= 2020) & (yrs <= 2022) & ~z
def ols(names):
    X = np.column_stack([F[n][fit] for n in names]); y = yr[fit]
    w = np.linalg.lstsq(X.T @ X, X.T @ y, rcond=None)[0]
    return dict(zip(names, w))
def show(nm, x):
    r = {wn: G.ev(np.where(z, 0.0, x), wn) for wn in W}
    print("%-30s " % nm + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
    print("%-30s " % "   vs 1610" + " | ".join("%s %+.4f/%+.3f/%+.1f/%+.3f/%+.4f" % (wn, *[r[wn][m] - ref[wn][m] for m in MET]) for wn in W), flush=True)
EQ = {
    "A0 段反转": {"D_R": -1},
    "A1 段反转去跳跃": {"D_NOJ": -1},
    "A2 段反转扣同伴": {"D_IDIO": -1},
    "A3 放量+缩量反转、跳跃延续": {"D_HV": -1, "D_LV": -1, "D_JUMP": 1},
    "A4 段 + 当日累计（都去跳跃）": {"D_NOJ": -1, "C_NOJ": -1},
    "C0 今日特质反转 P_CUM": {"P_CUM": 1},
    "C1 阶梯等权（今日、跳空、昨日、5 日）": {"P_CUM": 1, "P_GAP": 1, "P_DAY": 1, "P_5D": 1},
    "C2 阶梯 + 20 日": {"P_CUM": 1, "P_GAP": 1, "P_DAY": 1, "P_5D": 1, "P_20D": 1},
    "AC 段去跳跃 + 阶梯": {"D_NOJ": -1, "P_CUM": 1, "P_GAP": 1, "P_DAY": 1, "P_5D": 1},
}
for nm, wd in EQ.items():
    show(nm, sum(s * F[k] for k, s in wd.items()) / len(wd))
SETS = {"OLS-A（段与当日的三块 + 同伴段）": ["D_JUMP", "D_HV", "D_LV", "C_JUMP", "C_HV", "C_LV", "P_SEG"],
        "OLS-C（阶梯）": ["P_CUM", "P_GAP", "P_DAY", "P_5D", "P_20D"],
        "OLS-AC（全部）": ["D_JUMP", "D_HV", "D_LV", "C_JUMP", "C_HV", "C_LV", "P_SEG", "P_CUM", "P_GAP", "P_DAY", "P_5D", "P_20D"]}
for nm, names in SETS.items():
    w = ols(names)
    print("%s 权重（2020–22）：" % nm + "，".join("%s %+.3f" % (k, v) for k, v in w.items()), flush=True)
    show(nm, sum(v * F[k] for k, v in w.items()))
