# -*- coding: utf-8 -*-
"""c93（10-20 晚）= c91 的导出版：把 1610／1603 + 主题的几种混合导出 2024 因子值，送评分机估分。原 c91 说明：集中优化第一步（10-20 晚）：纯方向主题"流动性冲击回吐"（订单流与特质冲击）与 1610／1603（可解释的线性合成）合并。
两者信息来源不同，看能否五项都好于 1610（公榜 0.891）。主题 THEME：
  快 = 平均(波动标准化特质段反转 VN_IDIO、特质放量／跳跃、特质主动买压、特质盘口订单流) 平滑 α；
  + 0.25·特质 VWAP 偏离（平滑 0.3）+ 0.4·委托结构（深档笔数 + 一档规模比，平滑 0.3）+ 0.25·(特质日内反转 + 订单流常态)
合成：底座 + w·K·THEME（K = 底座的尺度），w ∈ {0.05, 0.1, 0.2, 0.3, 0.5}；五窗口五项（★ = 不差）。"""
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, CLS, CP, build, MET, OUT
c = G.c; z = G.z930; did = c.date_id; slot = c.slot_of_date[did]
U.DEAL = {"Y_sd5", "X_n5d"}
W = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
K10, K03 = 0.2774 / 0.289, 0.655 / 0.289
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask=None):
    x = np.where((np.isfinite(x) & (x != 0)) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
sm = lambda x, a: H.ewma_col(x, c.inst, a)                                               # noqa: E731
DR, PS = raw("D_R"), raw("P_SEG"); mP = (DR != 0) & (PS != 0)
ID = np.where(mP, DR - PS, np.nan)
order = np.lexsort((pd.to_datetime(c.key_date).values.astype(np.int64), c.inst))
s = pd.Series(np.where(z, np.nan, ID)[order])
sd = s.groupby(c.inst[order]).transform(lambda q: q.rolling(140, min_periods=40).std().shift(1)).to_numpy()
sig = np.empty_like(sd); sig[order] = sd
FAST = np.mean([-rk(ID / sig), -rk(raw("D_HV") + raw("D_JUMP") - PS, mP), -rk(raw("PD_MS_OFI")), -rk(raw("PD_CF_OFI3"))], axis=0)
REST = 0.25 * sm(-rk(raw("PD_AN_VWAP")), 0.3) + 0.4 * sm(np.mean([rk(raw("QL_NIMB23")), -rk(raw("OQ_SIZE"))], axis=0), 0.3) \
    + 0.25 * (rk(raw("P_CUM")) + rk(raw("CF_OFI3_20")))
def ok(d):
    return d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
BASES = {"1610": ({wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in W.items()}, K10),
         "1603": ({wn: np.where(z, 0.0, build("1603", ix, drop_deal=True)) for wn, ix in W.items()}, K03)}
from g_go import inst_codes
kd = pd.to_datetime(c.key_date); m24 = (kd.year == 2024); ins = inst_codes(c)
MIX = [("1610", 0.6, 0.2), ("1610", 0.6, 0.3), ("1610", 0.7, 0.2), ("1610", 0.7, 0.3), ("1603", 0.7, 0.05), ("1603", 0.7, 0.1), ("1610", 0.6, 0.0)]
THS = {}
for b, a, w in MIX:
    if a not in THS:
        THS[a] = np.where(z, 0.0, sm(FAST, a) + REST)
    Bb, K = BASES[b]
    x = Bb["2024"] + w * K * THS[a]
    o = pd.DataFrame({"date": kd[m24], "instrument": ins[np.flatnonzero(m24)], "factor": x[m24]}).drop_duplicates(["date", "instrument"], keep="last")
    o.to_parquet(OUT / ("val_M_%s_a%d_w%02d_2024.parquet" % (b, a * 10, w * 100)), index=False)
    print("导出", b, a, w, flush=True)
