# -*- coding: utf-8 -*-
"""因子榜校准：本地评分（平台口径，localeval.evaluate）与公榜分数的秩相关。
snap = 同一时刻的公榜快照（2026-10-12，16 个因子）；plat = factor_lib 里登记的全部已知分数（时点不一）。
注意：本地因子值来自 30 分钟因子库，与平台 SQL 的因子值并不完全相同（reports/12 第三节）。
用法：[ONLY="90 104"] bash 工具/本地评分/remote.sh e2e/scripts/rank_cal.py [start] [end]
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"   # 每个进程单线程，并行靠多进程（fork 共享面板）
import multiprocessing as mp  # noqa: E402

import pandas as pd  # noqa: E402

sys.path.insert(0, "工具/本地评分")
import localeval as le  # noqa: E402
from factor_lib import FACTORS, add_components  # noqa: E402

start, end = (sys.argv[1:3] + ["2023-01", "2024-12"])[:2] if len(sys.argv) > 1 else ("2023-01", "2024-12")
SNAP = {"107": 0.78637, "106": 0.69983, "105": 0.79659, "104": 0.81874, "103": 0.81090, "102": 0.35366, "101": 0.65758,
        "100": 0.64259, "99": 0.64940, "98": 0.68995, "97": 0.69881, "96": 0.73765, "93": 0.72129, "92": 0.72300,
        "91": 0.71652, "90": 0.76303}
OUT_DIR = os.path.join("e2e", "runs", "analysis")
NJ = int(os.environ.get("NJ", "16"))
MET = ["ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover"]
p = add_components(le.build_panel(le.load_bars(start, end)))


def work(i):
    fid, name, plat, fn, note = FACTORS[i]
    m = le.evaluate(fn(p), p)
    print(fid, name, flush=True)
    return {"id": fid, "name": name, "plat": plat, "snap": SNAP.get(fid), **{k: m[k] for k in MET}}


if __name__ == "__main__":
    ONLY = os.environ.get("ONLY", "").split()
    idx = [i for i, f in enumerate(FACTORS) if not ONLY or f[0] in ONLY]
    with mp.get_context("fork").Pool(NJ, maxtasksperchild=4) as pool:
        t = pd.DataFrame(pool.map(work, idx, chunksize=1)).set_index("id")
    os.makedirs(OUT_DIR, exist_ok=True)
    t.to_csv(os.path.join(OUT_DIR, f"rank_cal_{start}_{end}.csv"))
    t["comp"] = le.composite(t)
    out = []
    for tgt in ("snap", "plat"):
        k = t.dropna(subset=[tgt])
        rec = {"target": tgt, "n": len(k), "comp": k["comp"].rank().corr(k[tgt].rank())}
        for c in MET:
            rec[c] = (-k[c] if c == "turnover" else k[c]).rank().corr(k[tgt].rank())
        out.append(rec)
    pd.set_option("display.width", 250)
    print(f"== 本地指标与公榜分数的秩相关（{start}..{end}）")
    print(pd.DataFrame(out).round(3).to_string(index=False))
    s = t.dropna(subset=["snap"]).copy()
    s["rk_plat"] = s["snap"].rank(ascending=False).astype(int)
    s["rk_local"] = s["comp"].rank(ascending=False).astype(int)
    print("== 快照 16 个因子：公榜名次 vs 本地名次")
    print(s.sort_values("snap", ascending=False)[["name", "snap", "comp", "rk_plat", "rk_local"] + MET].round(4).to_string())
