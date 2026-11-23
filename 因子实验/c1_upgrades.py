# -*- coding: utf-8 -*-
"""Claude 夜间 C1：今天已交三个线性方向（1602／1603／1601）的升级版筛选，口径 engine.Core.evaluate（= 平台）。
公榜已证实的杠杆：
  - 根权重剂量（1378／1379：用少量 IC 换换手，公榜 +0.003／+0.005）；
  - IC_IR 与换手 1:1 兑现，根的 IC 增量在公榜上 ≈ 0；
  - 依赖成交笔数的根（N_dir、Y_sd5、X_n5d）在平台上为零 → 本脚本一律先把它们去掉，作为"平台一致"的基准。
变体：剂量 k（根这一块 ×k）、快根平滑（根轴先做 EWMA）、日内不变锚 CLOSEPOS20、收盘段季节性 CLS20 只全天不加份。
用法（personal）：~/jyc_scorer/.venv/bin/python 因子实验/c1_upgrades.py
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
from g_go import GO, WINS, IXW, SPEC  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
W1363 = {"L_lab10": 0.20, "M_q_lin": 0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": 0.10,
         "Y_sd5": 0.10, "X_n5d": -0.10, "Z_mfr60": -0.15, "Z_mfr90": -0.10}
SMALL = {"k1381": 0.02, "k1385": 0.02, "k1406": 0.05}
DEAL = {"N_dir", "Y_sd5", "X_n5d"}                                  # 平台上为零的成交笔数类根
D2 = {"L_lab10": 0.0614, "M_q_lin": 0.0139, "N_dir": 0.0073, "W_vdir": -0.0288, "W_zero": 0.0939, "Y_sd5": 0.0085,
      "X_n5d": -0.0170, "Z_mfr60": -0.1583, "Z_mfr90": 0.0076, "s155": -0.1825, "s139": 0.3831, "s161": 0.3000,
      "k1381": 0.0177, "k1385": 0.0275, "k1406": 0.0063}
G4 = {k: v for k, v in json.load(open(OUT / "g4_opt.json", encoding="utf-8"))["runs"]["3.0000"]["delta"].items() if abs(v) > 1e-12}
ROOTS = list(W1363)
MET = ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")

G = GO()
c = G.c
z930 = G.z930
slot = c.slot_of_date[c.date_id]


def rank_npy(name):
    x = np.load(OUT / ("h1_%s.npy" % name))
    r = np.nan_to_num(O.rc_of(np.asarray(x, float), c.date_id, c.n_dates), nan=0.0)
    r[z930] = 0.0
    return r


CLS = rank_npy("CLS20")
CP = rank_npy("CLOSEPOS20")
AXS = {}


def ax(lab, ix, alpha=None):
    """根轴；alpha=None 用默认（L_lab10 为 0.75，其余不平滑），否则对该根先做 EWMA α。"""
    if alpha is None:
        return G.axis(lab, ix)
    k = (lab, ix, alpha)
    if k not in AXS:
        n1, n21, n20, al0 = SPEC[lab]
        nm = (n1, n21, n20)[ix]
        rng = ("2020-01-01", "2024-12-31") if ix == 0 else ("2020-01-01", "2022-12-31")
        a = max(alpha, float(al0 or 0.0))
        AXS[k] = np.nan_to_num(O.axis(c, nm, *rng, alpha=a), nan=0.0)
    return AXS[k]


def build(kind, ix, dose=1.0, smooth=None, cp=0.0, cls_all=None, cls_boost=None, drop_deal=True):
    """kind ∈ {1602, 1603, 1601}；返回原始值合成（09:30 置 0）。"""
    roots = [r for r in ROOTS if not (drop_deal and r in DEAL)]
    if kind == "1602":
        B = np.asarray(G.B[("s161", ix)], float).copy()
        for r in roots:
            B += dose * W1363[r] * ax(r, ix, smooth if r != "L_lab10" else None)
        for k, w in SMALL.items():
            B += w * G.axis(k, ix)
        sd = 0.2774
    else:
        base = np.asarray(G.B[("1363", ix)], float).copy()
        head = base - sum(W1363[r] * G.axis(r, ix) for r in ROOTS)        # 490 头部（近似：1363 减去它内部的九根）
        delta = G4 if kind == "1603" else D2
        B = head.copy()
        for r in roots:
            B += dose * (W1363[r] + delta.get(r, 0.0)) * ax(r, ix, smooth if r != "L_lab10" else None)
        for k, w in delta.items():
            if k in ROOTS:
                continue
            if k in ("s155", "s139", "s161"):
                B += w * G.axis(k, ix)
            else:
                B += w * G.axis(k, ix)
        sd = 0.655 if kind == "1603" else 0.544
        a, b = (0.115, 0.23) if kind == "1603" else (0.094, 0.188)
        a = a if cls_all is None else cls_all
        b = b if cls_boost is None else cls_boost
        B += a * CLS + b * CLS * (slot == 1430)
    if cp:
        B += cp * sd / 0.289 * CP
    if kind == "1602" and cls_all:
        B += cls_all * CLS
    B[z930] = 0.0
    return B


def evalB(Bs):
    out = {}
    for wn, _, _ in WINS:
        m = G.ev(Bs[IXW[wn]], wn)
        out[wn] = {k: m[k] for k in MET}
    return out


def est(d):
    """公榜估计的差（两种定价取均值）：仅换手 −0.37／单位；含 152 的回归 3.94·IC + 0.132·IC_IR − 0.217·换手。根的 IC 增量按 0 兑现。"""
    a = -0.37 * d["turnover"]
    b = 0.132 * d["ic_ir"] - 0.217 * d["turnover"]
    return 0.5 * (a + b)


VARS = []
for kind in ("1602", "1603", "1601"):
    VARS.append((kind, "已交版（含成交笔数根）", dict(drop_deal=False)))
    VARS.append((kind, "平台一致（去成交笔数根）", dict()))
    for dose in (1.25, 1.5, 2.0):
        VARS.append((kind, "剂量 ×%g" % dose, dict(dose=dose)))
    for sm in (0.5, 0.75):
        VARS.append((kind, "快根平滑 α%g" % sm, dict(smooth=sm)))
        VARS.append((kind, "剂量 ×1.5 + 平滑 α%g" % sm, dict(dose=1.5, smooth=sm)))
    VARS.append((kind, "+CLOSEPOS20 0.05", dict(cp=0.05)))
    VARS.append((kind, "剂量 ×1.5 + CLOSEPOS20 0.05", dict(dose=1.5, cp=0.05)))
    if kind == "1602":
        VARS.append((kind, "+CLS20 全天 0.03", dict(cls_all=0.03)))

if __name__ == "__main__":
    ref = {}
    for kind, name, kw in VARS:
        Bs = {ix: build(kind, ix, **kw) for ix in (0, 1, 2)}
        r = evalB(Bs)
        if "平台一致" in name:
            ref[kind] = r
        base = ref.get(kind)
        line = "%-5s %-24s" % (kind, name)
        for wn in ("2023", "2024", "2021", "2022", "2020"):
            m = r[wn]
            if base is None or name.startswith("平台一致"):
                line += " | %s %.4f/%.3f/%.1f/%.3f/%.4f" % (wn, m["ic_mean"], m["ic_ir"], m["ls_sharpe"], m["stress"], m["turnover"])
            else:
                d = {k: m[k] - base[wn][k] for k in MET}
                line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f e%+.4f" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], est(d))
        print(line, flush=True)
