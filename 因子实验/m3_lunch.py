# -*- coding: utf-8 -*-
"""E6（10-19）：午休缺口 LG = 13:00 开盘 ÷ 11:30 收盘 − 1（A 股的"午间小隔夜"），只对 13:30／14:00／14:30 截面有值，上午为空。
保存 vc_tmp/h1_LG.npy；打印各下午截面的秩 IC（两个方向看符号），以及按锚的方式叠到 1610 上的五窗口五项。"""
import numpy as np
import pandas as pd
import h1_season as H1
import c1_upgrades as U
import d2_optimize as O
from c1_upgrades import G, CLS, CP, build, MET
c = G.c; z = G.z930; slot = c.slot_of_date[c.date_id]
U.DEAL = {"Y_sd5", "X_n5d"}
t = pd.read_parquet(H1.CACHE); t["day"] = pd.to_datetime(t["day"])
W = H1.wide(t); del t
lg = (W["open"]["1330"] / W["close"]["1130"].where(W["close"]["1130"] > 0) - 1.0).replace([np.inf, -np.inf], np.nan)
B = pd.DataFrame({s0: (lg.to_numpy() if s0 in ("1330", "1400", "1430") else np.full(len(lg), np.nan)) for s0 in H1.CS}, index=lg.index)
x = H1.align(c, B); x[z] = np.nan
np.save(H1.OUT / "h1_LG.npy", np.nan_to_num(x, nan=0.0))
r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0); r[z] = 0.0
kd = pd.to_datetime(c.key_date)
for yr in (2020, 2021, 2022, 2023, 2024):
    line = "%d" % yr
    for s0 in (1330, 1400, 1430):
        msk = np.asarray((kd.year == yr) & (slot == s0) & np.isfinite(x) & np.isfinite(c.y))
        ic = np.nanmean(c.per_date_rank_corr(r, np.nan_to_num(c.y), msk))
        line += " | %d IC %+.4f" % (s0, ic)
    print("LG " + line, flush=True)
K10 = 0.2774 / 0.289
WW = {"2023": 0, "2024": 0, "2021": 1, "2022": 1, "2020": 2}
base = {wn: np.where(z, 0.0, build("1602", ix, drop_deal=True) + 0.05 * K10 * CLS + 0.10 * K10 * CLS * (slot == 1430) + 0.075 * K10 * CP) for wn, ix in WW.items()}
ref = {wn: G.ev(base[wn], wn) for wn in WW}
for sg in (1, -1):
    for w in (0.025, 0.05, 0.10):
        line, st = "1610 %+.3f·LG" % (sg * w), 0
        for wn in WW:
            q = G.ev(base[wn] + sg * w * K10 * r, wn)
            d = {m: q[m] - ref[wn][m] for m in MET}
            ok = d["ic_mean"] >= -0.00005 and d["ic_ir"] >= 0 and d["ls_sharpe"] >= 0 and d["stress"] >= 0 and d["turnover"] <= 0
            st += ok
            line += " | %s %+.5f/%+.4f/%+.2f/%+.4f/%+.4f%s" % (wn, d["ic_mean"], d["ic_ir"], d["ls_sharpe"], d["stress"], d["turnover"], "★" if ok else "")
        print(line + "  %d/5" % st, flush=True)
