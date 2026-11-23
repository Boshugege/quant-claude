# -*- coding: utf-8 -*-
"""新方向 D（10-19 傍晚）：按经济机制分组的等风险合成——不用优化器，分组与符号都由经济先验定，可解释性来自构造本身。
组（组内各分量截面秩等权，组合成后逐截面标准化）：
  锚点反转  s161 头部（同伴比较的价格锚点反转）、490 头部（多尺度反转阶梯）
  流动性提供 Z_mfr60、Z_mfr90、W_vdir（资金流 / 量价同向，过度买压后回吐，均取反向）
  知情交易  M_q_lin（聪明钱 +）、N_dir（当日大单方向 −）、Y_sd5（近 5 日单笔额离散度 +）、X_n5d（近 5 日大单方向 −）
  价格摩擦  W_zero（零收益分钟占比 +）
  季节性    L_lab10（同时段季节性 +）、CLS20（收盘段季节性 +）
  收盘行为  CLOSEPOS20（收盘位置 +）、k1406（前日收盘竞价冲击 +）
  分布形态  k1381（日内峰度反向 +）、k1385（近 5 日偏度反向 +）
  盘口深层  BK_SLOPEI（斜率失衡 +）、BK_CONS（深度消耗率 −）
组间配权：a 八组等风险；b 锚点反转 50%、其余七组平分 50%；c 同 b 但去掉盘口组。与 1610（平台口径）五窗口五项比较。"""
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, W1363, MET, OUT, ROOTS
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]; did = c.date_id
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10 = 0.2774 / 0.289
def rk(x):
    r = np.nan_to_num(O.rc_of(np.asarray(x, float), did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
def npy(nm):
    x = np.load(OUT / ("h1_%s.npy" % nm)).astype(float); x[x == 0] = np.nan
    return rk(x)
def zs(x):
    s = pd.Series(x); g = s.groupby(did)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()
BK = {"BK_SLOPEI": npy("BK_SLOPEI"), "BK_CONS": npy("BK_CONS")}
def groups(ix):
    head490 = np.asarray(G.B[("1363", ix)], float) - sum(W1363[r] * G.axis(r, ix) for r in ROOTS)
    A = lambda r: G.axis(r, ix)
    return {
        "锚点反转": [rk(G.B[("s161", ix)]), rk(head490)],
        "流动性提供": [-A("Z_mfr60"), -A("Z_mfr90"), -A("W_vdir")],
        "知情交易": [A("M_q_lin"), -A("N_dir"), A("Y_sd5"), -A("X_n5d")],
        "价格摩擦": [A("W_zero")],
        "季节性": [A("L_lab10"), CLS],
        "收盘行为": [CP, G.axis("k1406", ix)],
        "分布形态": [G.axis("k1381", ix), G.axis("k1385", ix)],
        "盘口深层": [BK["BK_SLOPEI"], -BK["BK_CONS"]],
    }
def compose(ix, scheme):
    gs = {k: zs(np.mean(v, axis=0)) for k, v in groups(ix).items()}
    if scheme == "c":
        gs.pop("盘口深层")
    if scheme == "a":
        w = {k: 1.0 / len(gs) for k in gs}
    else:
        w = {k: (0.5 if k == "锚点反转" else 0.5 / (len(gs) - 1)) for k in gs}
    return np.where(z, 0.0, sum(w[k] * gs[k] for k in gs))
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
if __name__ == "__main__":
  U.DEAL = {"Y_sd5", "X_n5d"}
  ref = {wn: G.ev(np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP), wn) for wn, ix in W.items()}
  print("1610 " + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, *[ref[wn][m] for m in MET]) for wn in W), flush=True)
  for sch in ("a", "b", "c"):
      line, st = "D-%s" % sch, 0
      for wn, ix in W.items():
          r = G.ev(compose(ix, sch), wn); d = {m: r[m] - ref[wn][m] for m in MET}
          st += ok(d)
          line += " | %s %.4f/%.3f/%.1f/%.3f/%.4f（%+.4f/%+.3f）" % (wn, r["ic_mean"], r["ic_ir"], r["ls_sharpe"], r["stress"], r["turnover"], d["ic_mean"], d["ic_ir"])
      print(line + "  相对 1610 %d/5" % st, flush=True)
