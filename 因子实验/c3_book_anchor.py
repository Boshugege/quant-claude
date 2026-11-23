# -*- coding: utf-8 -*-
"""Claude 夜间 C3：盘口结构的"日内不变锚"（新信息源，不用成交笔数）。
思路：CLS20／CLOSEPOS20 证明"当日不变 + 本身有 IC"的分量能同时提 IC_IR、降换手。盘口的委托结构是另一类信息：
  OSZ  买方平均每笔挂单量 ÷ 卖方（1–3 档合计，对数）——大资金在哪一侧排队；
  OIM  委托笔数不平衡 (买笔数 − 卖笔数) ÷ 合计；
  DIM  挂单量不平衡 (买量 − 卖量) ÷ 合计。
每个先在当日 8 个半小时段末快照上取均值，再取过去 N 日（5／20）均值并错开 1 日 → 当日不变。
数据：评测机本地 30 分钟 bar（含 1–3 档挂单量与委托笔数，段末快照）。
评估：单独（全截面）五项；叠到 1606 的平台一致版（剂量 ×1.5 + CLOSEPOS20）上，权重 0.03／0.05（秩口径，按底座截面标准差换算）。
用法（personal，空闲内存 ≥ 15 GB 时）：~/jyc_scorer/.venv/bin/python 因子实验/c3_book_anchor.py
"""
import glob
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
from c1_upgrades import G, build, MET  # noqa: E402
from g_go import WINS, IXW  # noqa: E402

c = G.c
z930 = G.z930
SEGS = ["1000", "1030", "1100", "1130", "1330", "1400", "1430", "1500"]
cols = ["date", "instrument_id"] + ["%s_%s%d" % (s, k, i) for s in ("bid", "ask") for k in ("volume", "num_orders") for i in (1, 2, 3)]
parts = []
for f in sorted(glob.glob("/home/user/jyc_scorer/data/bar30m/*.feather")):
    t = pd.read_feather(f, columns=cols)
    hm = t["date"].dt.strftime("%H%M")
    t = t[hm.isin(SEGS)]
    bv = t[["bid_volume1", "bid_volume2", "bid_volume3"]].sum(1).astype(float)
    av = t[["ask_volume1", "ask_volume2", "ask_volume3"]].sum(1).astype(float)
    bn = t[["bid_num_orders1", "bid_num_orders2", "bid_num_orders3"]].sum(1).astype(float)
    an = t[["ask_num_orders1", "ask_num_orders2", "ask_num_orders3"]].sum(1).astype(float)
    with np.errstate(all="ignore"):
        osz = np.log((bv / bn.where(bn > 0)) / (av / an.where(an > 0)))
        oim = (bn - an) / (bn + an).where((bn + an) > 0)
        dim = (bv - av) / (bv + av).where((bv + av) > 0)
    d = pd.DataFrame({"iid": t["instrument_id"].to_numpy(), "day": t["date"].dt.normalize().to_numpy(),
                      "OSZ": osz.to_numpy(), "OIM": oim.to_numpy(), "DIM": dim.to_numpy()})
    parts.append(d.replace([np.inf, -np.inf], np.nan).groupby(["iid", "day"]).mean())
D = pd.concat(parts).sort_index()
del parts
print("日表", len(D), flush=True)
feats = {}
g = D.groupby(level=0, sort=False)
for col in ("OSZ", "OIM", "DIM"):
    for N in (5, 20):
        m = g[col].rolling(N, min_periods=max(N // 2, 3)).mean()
        m.index = m.index.droplevel(0)
        feats["%s%d" % (col, N)] = m.reindex(D.index).groupby(level=0, sort=False).shift(1)
F = pd.DataFrame(feats).reset_index()
need = pd.DataFrame({"row": np.arange(c.n_rows), "iid": c.inst.astype(np.int64),
                     "day": pd.to_datetime(c.day_of_date[c.date_id]).values})
M = need.merge(F, on=["iid", "day"], how="left").sort_values("row")
R = {}
for k in feats:
    x = M[k].to_numpy(float)
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
    r[z930] = 0.0
    R[k] = r


def ev5(Bs):
    return {wn: {m: G.ev(Bs[IXW[wn]], wn)[m] for m in MET} for wn, _, _ in WINS}


print("== 单独（全截面）IC／IC_IR／换手", flush=True)
for k, r in R.items():
    res = ev5({0: r, 1: r, 2: r})
    print("%-6s %s" % (k, " | ".join("%s %+.4f/%+.3f/%.3f" % (wn, res[wn]["ic_mean"], res[wn]["ic_ir"], res[wn]["turnover"])
                                    for wn in ("2023", "2024", "2021", "2022", "2020"))), flush=True)
base = {ix: build("1603", ix, dose=1.5, cp=0.05) for ix in (0, 1, 2)}
rb = ev5(base)
print("== 叠到 1606 平台一致版（剂量 ×1.5 + CLOSEPOS20）：Δ IC／IC_IR／多空／压力／换手", flush=True)
for k, r in R.items():
    for sgn in (1, -1):
        for w in (0.03, 0.05):
            Bs = {ix: base[ix] + sgn * w * 0.655 / 0.289 * r for ix in (0, 1, 2)}
            res = ev5(Bs)
            line = "%-6s %+.2f" % (k, sgn * w)
            for wn in ("2023", "2024", "2021", "2022", "2020"):
                d = {m: res[wn][m] - rb[wn][m] for m in MET}
                line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"])
            print(line, flush=True)
