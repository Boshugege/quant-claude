# -*- coding: utf-8 -*-
"""主题"流动性冲击回吐"第三轮（10-20）：c83 的前沿估分停在 0.82～0.83，要抬信号强度（IC）而不是继续用 IC 换换手。
  快：F4 = 四个特质快信号（特质段反转、特质放量／跳跃、特质主动买压、特质盘口订单流）的截面秩平均，平滑 α ∈ {0.6, 0.7}
  中：VW = −秩(VWAP 偏离)（本股 AN_VWAP 或特质 PD_AN_VWAP，平滑 0.3），权重 wv ∈ {0, 0.25, 0.5}
  委托：Q = 秩(第 2、3 档笔数失衡) − 秩(一档规模比) 的平均（平滑 0.3），权重 wq ∈ {0.25, 0.4}
  慢：0.25·(PCUM + FLOW20)，另加 we·(价差常态 + 微观价格常态 + 笔数失衡常态)，we ∈ {0, 0.15}
打印 2023／2024；按 2024 的 IC_IR − 1.5·换手 排序导出前 8 个的 2024 因子值（送评分机估公榜）。"""
import os
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
mD = raw("D_R") != 0; PS = raw("P_SEG"); mP = mD & (PS != 0)
F4 = np.mean([-rk(raw("D_R") - PS, mP), -rk(raw("D_HV") + raw("D_JUMP") - PS, mP), -rk(raw("PD_MS_OFI")), -rk(raw("PD_CF_OFI3"))], axis=0)
SM = {a: H.ewma_col(F4, c.inst, a) for a in (0.6, 0.7)}
VW = {"own": H.ewma_col(-rk(raw("AN_VWAP")), c.inst, 0.3), "idio": H.ewma_col(-rk(raw("PD_AN_VWAP")), c.inst, 0.3)}
Q = H.ewma_col(np.mean([rk(raw("QL_NIMB23")), -rk(raw("OQ_SIZE"))], axis=0), c.inst, 0.3)
SLOW = rk(raw("P_CUM")) + rk(raw("CF_OFI3_20"))
EXT = rk(raw("AN_SPRD_20")) + rk(raw("AN_MICRO_20")) + rk(raw("QL_NIMBA_20"))
W = ("2023", "2024")
res = []
ONLY = set(x for x in os.environ.get("ONLY_TAGS", "").split(",") if x)          # 只算并导出指定组合
for a in (0.6, 0.7):
    for vn in ("own", "idio"):
        for wv in (0.0, 0.25, 0.5):
            if wv == 0 and vn == "idio":
                continue
            for wq in (0.25, 0.4):
                for we in (0.0, 0.15):
                    x = np.where(z, 0.0, SM[a] + wv * VW[vn] + wq * Q + 0.25 * SLOW + we * EXT)
                    r = {wn: G.ev(x, wn) for wn in W}
                    tag = "a%d_%s_v%02d_q%02d_e%02d" % (a * 10, vn, wv * 100, wq * 100, we * 100)
                    if ONLY and tag not in ONLY:
                        continue
                    print("%-26s " % tag + " | ".join("%s %.4f/%.3f/%.1f/%.3f/%.3f" % (wn, *[r[wn][m] for m in MET]) for wn in W), flush=True)
                    res.append((r["2024"]["ic_ir"] - 1.5 * r["2024"]["turnover"], tag, x))
res.sort(key=lambda t: -t[0])
kd = pd.to_datetime(c.key_date); m = (kd.year == 2024); ins = inst_codes(c)
for _, tag, x in (res if ONLY else res[:int(os.environ.get("TOPN", 8))]):
    o = pd.DataFrame({"date": kd[m], "instrument": ins[np.flatnonzero(m)], "factor": x[m]}).drop_duplicates(["date", "instrument"], keep="last")
    o.to_parquet(OUT / ("val_G_%s_2024.parquet" % tag), index=False)
    print("导出", tag, flush=True)
