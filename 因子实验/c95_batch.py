# -*- coding: utf-8 -*-
"""批量筛选（10-20 晚，用户："一批一批做很多，同时跑，选可行的改"）：在 personal 上生成一批候选的 2024 因子值，直接提交本机评分机（kind=values），
评分机给五项与估公榜分。BATCH 环境变量选批次（b1 = 单信号最好形式 + 主题变体 + 与 1610／1603 混合）。"""
import os
import subprocess
import numpy as np
import pandas as pd
import c1_upgrades as U
import d2_optimize as O
import h2_robust as H
from c1_upgrades import G, CLS, CP, build, OUT
from g_go import inst_codes
BATCH = os.environ.get("BATCH", "b1")
INC = os.path.expanduser("~/jyc_scorer/incoming/%s" % BATCH); os.makedirs(INC, exist_ok=True)
c = G.c; z = G.z930; did = c.date_id; slot = c.slot_of_date[did]
U.DEAL = {"Y_sd5", "X_n5d"}
K10, K03 = 0.2774 / 0.289, 0.655 / 0.289
raw = lambda nm: np.load(OUT / ("h1_%s.npy" % nm)).astype(float)
def rk(x, mask=None):
    x = np.where((np.isfinite(x) & (x != 0)) if mask is None else mask, x, np.nan)
    r = np.nan_to_num(O.rc_of(x, did, c.n_dates), nan=0.0); r[z] = 0.0
    return r
sm = lambda x, a: H.ewma_col(x, c.inst, a) if a > 0 else x                               # noqa: E731
kd = pd.to_datetime(c.key_date); m24 = (kd.year == 2024); ins = inst_codes(c)[np.flatnonzero(m24)]
SUBMIT = os.environ.get("SUBMIT", "1") == "1"                                              # 0 = 只导出（在 macstudio 上生成，再传到 personal 提交）
if not SUBMIT:
    INC = str(OUT / ("batch_%s" % BATCH)); os.makedirs(INC, exist_ok=True)
def submit(name, x):
    x = np.where(z, 0.0, np.nan_to_num(x, nan=0.0))
    f = os.path.join(INC, "%s.parquet" % name)
    pd.DataFrame({"date": kd[m24], "instrument": ins, "factor": x[m24]}).drop_duplicates(["date", "instrument"], keep="last").to_parquet(f, index=False)
    if not SUBMIT:
        print("导出", name, flush=True); return
    r = subprocess.run(["curl", "-s", "-F", "files=@%s" % f, "-F", "kind=values", "-F", "start=2024-01", "-F", "end=2024-12",
                        "-F", "name=%s_%s" % (BATCH, name), "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print("提交", name, r.stdout.strip()[:90], flush=True)
# ---- 分量
DR, PS = raw("D_R"), raw("P_SEG"); mP = (DR != 0) & (PS != 0)
ID = np.where(mP, DR - PS, np.nan)
order = np.lexsort((kd.values.astype(np.int64), c.inst))
sd = pd.Series(np.where(z, np.nan, ID)[order]).groupby(c.inst[order]).transform(lambda q: q.rolling(140, min_periods=40).std().shift(1)).to_numpy()
sig = np.empty_like(sd); sig[order] = sd
R = {"VNIDIO": -rk(ID / sig), "IDSEG": -rk(ID), "IDHVJ": -rk(raw("D_HV") + raw("D_JUMP") - PS, mP),
     "ITICK": -rk(raw("PD_MS_OFI")), "ICOFI": -rk(raw("PD_CF_OFI3")), "ITSBIG": -rk(raw("PD_TS_BIG")),
     "IVWAP": -rk(raw("PD_AN_VWAP")), "VWAP": -rk(raw("AN_VWAP")), "IRPOS": -rk(raw("PD_AN_RPOS")),
     "PCUM": rk(raw("P_CUM")), "N23": rk(raw("QL_NIMB23")), "NA": rk(raw("QL_NIMBA")), "S1": -rk(raw("OQ_SIZE")),
     "IN23": rk(raw("PD_QL_NIMB23")), "IS1": -rk(raw("PD_OQ_SIZE")),
     "FLOW20": rk(raw("CF_OFI3_20")), "SPRD20": rk(raw("AN_SPRD_20")), "MICRO20": rk(raw("AN_MICRO_20")), "NA20": rk(raw("QL_NIMBA_20"))}
F4 = np.mean([R["VNIDIO"], R["IDHVJ"], R["ITICK"], R["ICOFI"]], axis=0)
Q = sm(np.mean([R["N23"], R["S1"]], axis=0), 0.3)
IQ = sm(np.mean([R["IN23"], R["IS1"]], axis=0), 0.3)
SLOW = R["PCUM"] + R["FLOW20"]
EXT = R["SPRD20"] + R["MICRO20"] + R["NA20"]
def theme(a=0.6, wv=0.25, wq=0.4, ws=0.25, we=0.0, fast=F4, q=Q, vw="IVWAP"):
    return sm(fast, a) + wv * sm(R[vw], 0.3) + wq * q + ws * SLOW + we * EXT
CANDS = {}
if BATCH == "b1":
    for k, a in (("VNIDIO", 0.6), ("IDSEG", 0.6), ("IDHVJ", 0.6), ("ITICK", 0.6), ("ICOFI", 0.6), ("ITSBIG", 0.6), ("IVWAP", 0.3), ("VWAP", 0.3),
                 ("IRPOS", 0.3), ("PCUM", 0.0), ("N23", 0.3), ("S1", 0.3), ("FLOW20", 0.0), ("SPRD20", 0.0)):
        CANDS["s_%s_a%d" % (k, a * 10)] = sm(R[k], a)
    CANDS["t_A_a6"] = theme(0.6)
    CANDS["t_B_a7"] = theme(0.7)
    CANDS["t_C_tsbig"] = theme(0.6, fast=np.mean([R["VNIDIO"], R["IDHVJ"], R["ITICK"], R["ICOFI"], R["ITSBIG"]], axis=0))
    CANDS["t_D_vn_only"] = theme(0.6, wv=0.5, fast=R["VNIDIO"])
    CANDS["t_E_ext"] = theme(0.6, wv=0.5, we=0.15)
    CANDS["t_F_heavy"] = theme(0.65, wv=0.35, wq=0.5, ws=0.3)
    CANDS["t_G_iq"] = theme(0.6, q=IQ)
    CANDS["t_H_vwapq"] = 0.5 * sm(R["IVWAP"], 0.3) + 0.5 * Q
    B10 = np.where(z, 0.0, build("1602", 0, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP)
    B03 = np.where(z, 0.0, build("1603", 0, drop_deal=True))
    TA, TB = np.where(z, 0.0, theme(0.6)), np.where(z, 0.0, theme(0.7))
    CANDS["m_1610"] = B10
    CANDS["m_1603"] = B03
    for w in (0.05, 0.1, 0.2, 0.3, 0.5):
        CANDS["m_1610_A_w%02d" % (w * 100)] = B10 + w * K10 * TA
    for w in (0.2, 0.3):
        CANDS["m_1610_B_w%02d" % (w * 100)] = B10 + w * K10 * TB
    for w in (0.05, 0.1, 0.2):
        CANDS["m_1603_B_w%02d" % (w * 100)] = B03 + w * K03 * TB
if BATCH == "b2":                                                                          # 方向探索：每个都是一个有含义的新组合
    SKIP = -rk(raw("RC_FIRST25"))                                                           # 跳过段内最后 5 分钟的反转（N5：最近的冲击还在延续）
    POSI = np.mean([Q, R["FLOW20"], R["NA20"], R["MICRO20"]], axis=0)                      # 排队与订单流常态（正向信息）
    ANCH = np.mean([sm(R["IVWAP"], 0.3), R["PCUM"], sm(R["IRPOS"], 0.3)], axis=0)            # 日内锚定
    TSZ = sm(R["ITSBIG"], 0.6) + 0.5 * rk(raw("TS_SMALL"))                                  # 大单反转、小单延续
    for a in (0.5, 0.7):
        CANDS["d1_skip5_a%d" % (a * 10)] = sm(SKIP, a)
        CANDS["d2_shock_anchor_a%d" % (a * 10)] = sm(R["VNIDIO"], a) + sm(R["IVWAP"], 0.3)
    CANDS["d3_posinfo"] = POSI
    CANDS["d4_intraday_anchor"] = ANCH
    CANDS["d5_bigsmall"] = TSZ
    CANDS["d6_theme_skip"] = theme(0.6, fast=np.mean([SKIP, R["IDHVJ"], R["ITICK"], R["ICOFI"]], axis=0))
    CANDS["d7_theme_plus_posi"] = theme(0.6) + 0.3 * POSI
    CANDS["d8_theme_plus_anch"] = theme(0.6) + 0.3 * ANCH
    CANDS["d9_theme_plus_tsz"] = theme(0.6) + 0.3 * TSZ
    CANDS["d10_anchor_posi"] = ANCH + POSI
    CANDS["d11_shock_anchor_posi"] = sm(R["VNIDIO"], 0.6) + sm(R["IVWAP"], 0.3) + POSI
    CANDS["d12_theme_all"] = theme(0.6) + 0.2 * (POSI + ANCH + TSZ)
if BATCH == "b3":                                                                          # 主题细调 + 两个新方向
    TA = theme(0.6)
    for a in (0.55, 0.65, 0.75):
        CANDS["e_a%02d" % (a * 100)] = theme(a)
    for wv in (0.15, 0.35):
        CANDS["e_wv%02d" % (wv * 100)] = theme(0.6, wv=wv)
    for wq in (0.3, 0.5):
        CANDS["e_wq%02d" % (wq * 100)] = theme(0.6, wq=wq)
    for ws in (0.15, 0.35):
        CANDS["e_ws%02d" % (ws * 100)] = theme(0.6, ws=ws)
    SW = {1000: 1.2, 1030: 1.1, 1100: 1.0, 1130: 1.2, 1330: 1.0, 1400: 0.9, 1430: 0.7}       # 条件反转：开盘附近与 11:30 前更强、14:30 偏弱
    CANDS["e_slotw"] = TA * np.vectorize(lambda s_: SW.get(int(s_), 1.0))(slot)
    zt = (TA - TA.mean()) / (TA.std() + 1e-12)
    CANDS["e_tail"] = zt + 0.3 * zt * np.abs(zt)                                            # 强调极端值（极端反转更强）
    CANDS["e_dual"] = sm(R["VNIDIO"], 0.6) + sm(R["VNIDIO"], 0.9) + 0.25 * sm(R["IVWAP"], 0.3) + 0.4 * Q + 0.25 * SLOW   # 长短两种记忆
    dk = pd.Series(np.where(z, np.nan, F4)).groupby([c.inst, kd.normalize()]).transform(lambda q: q.ewm(alpha=0.4, adjust=False, ignore_na=True).mean()).to_numpy()
    CANDS["e_dayreset"] = np.nan_to_num(dk) + 0.25 * sm(R["IVWAP"], 0.3) + 0.4 * Q + 0.25 * SLOW   # 只在当天内平滑
if BATCH == "b4":                                                                          # 1610 + 主题的混合整体再平滑（压换手）
    B10 = np.nan_to_num(np.where(z, 0.0, build("1602", 0, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP))
    TA = np.where(z, 0.0, theme(0.6))
    for w in (0.0, 0.1, 0.2, 0.3):
        for a in (0.2, 0.35, 0.5):
            CANDS["f_1610_w%02d_s%02d" % (w * 100, a * 100)] = sm(B10 + w * K10 * TA, a)
if BATCH == "b5":                                                                          # 稀疏可解释：1610 + 少数几项（c92 定权里权重最大的几项），替代 12 项拟合
    B10 = np.nan_to_num(np.where(z, 0.0, build("1602", 0, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP))
    s10 = B10[~z].std()
    zz = lambda v: np.where(z, 0.0, v / (v[~z].std() + 1e-12))                              # noqa: E731
    P = {"Q": zz(Q), "IQ": zz(IQ), "N23": zz(sm(R["N23"], 0.3)), "S1": zz(sm(R["S1"], 0.3)), "SP": zz(R["SPRD20"]), "FL": zz(R["FLOW20"]),
         "TK": zz(-sm(R["ITICK"], 0.6)), "HV": zz(sm(R["IDHVJ"], 0.6)), "VW": zz(sm(R["IVWAP"], 0.3))}
    def mix(**w):
        return B10 / s10 + sum(v * P[k] for k, v in w.items())
    for w in (0.2, 0.3, 0.4):
        CANDS["g_q%02d" % (w * 100)] = mix(Q=w)
    CANDS["g_iq30"] = mix(IQ=0.3)
    CANDS["g_n23_30"] = mix(N23=0.3)
    CANDS["g_s1_30"] = mix(S1=0.3)
    CANDS["g_q30_sp15"] = mix(Q=0.3, SP=0.15)
    CANDS["g_q30_sp15_fl10"] = mix(Q=0.3, SP=0.15, FL=0.1)
    CANDS["g_q30_sp15_tk15"] = mix(Q=0.3, SP=0.15, TK=0.15)
    CANDS["g_q30_sp15_hv15"] = mix(Q=0.3, SP=0.15, HV=0.15)
    CANDS["g_q30_sp15_tk15_hv15"] = mix(Q=0.3, SP=0.15, TK=0.15, HV=0.15)
    CANDS["g_q30_sp15_tk15_hv15_vw10"] = mix(Q=0.3, SP=0.15, TK=0.15, HV=0.15, VW=0.1)
    CANDS["g_q40_sp20_tk20_hv15"] = mix(Q=0.4, SP=0.2, TK=0.2, HV=0.15)
    CANDS["g_q30_tk15"] = mix(Q=0.3, TK=0.15)
for nm, x in CANDS.items():
    submit(nm, x)
print("本批共", len(CANDS), flush=True)
