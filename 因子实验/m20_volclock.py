# -*- coding: utf-8 -*-
"""纯方向深挖 J8（10-20，跑在 personal 的 agent.slice）：成交量时钟下的反转。
固定 30 分钟窗口改为"最近相当于 N 分钟平均成交量的那一段"：今日至截面的平均每分钟成交量 × N 作为窗口量，
从截面往回累计成交量到这个量为止，窗口内的对数收益。活跃时窗口短、冷清时窗口长，按信息到达的节奏取样（事件时间）。
  VC_R10 / VC_R30 / VC_R60   N = 10 / 30 / 60 分钟
  VC_LEN30                    N = 30 时窗口实际跨的分钟数（对照：活跃度）
保存 vc_tmp/h1_VC_*.npy（原始值）。personal：python 用 ~/jyc_scorer/.venv，PYTHONPATH 指向 ~/quant-claude/因子实验（只读引用）。"""
import glob
import os
from multiprocessing import Pool
import numpy as np
import pandas as pd
import h1_season as H1
import root_eval as R

SRC = os.environ.get("SRC1M", "/srv/media/jyc_scorer/cpt_jyc_2026_e2e_bar1m/*.feather")
SLOTK = {"1000": 30, "1030": 60, "1100": 90, "1130": 120, "1330": 150, "1400": 180, "1430": 210}
NS = (10, 30, 60)


def one(f):
    d = pd.read_feather(f, columns=["date", "instrument_id", "close", "volume"])
    hm = (d["date"].dt.hour * 100 + d["date"].dt.minute).to_numpy()
    keep = ((hm > 930) & (hm <= 1130)) | ((hm > 1300) & (hm <= 1500))
    d, hm = d[keep], hm[keep]
    m = np.where(hm <= 1130, (hm // 100 - 9) * 60 + hm % 100 - 30, 120 + (hm // 100 - 13) * 60 + hm % 100) - 1
    day = d["date"].dt.normalize().to_numpy()
    pid = pd.MultiIndex.from_arrays([d["instrument_id"].to_numpy(np.int64), day])
    pairs, pi = pid.unique(), None
    pi = pairs.get_indexer(pid)
    P = len(pairs)
    lc = np.full((P, 240), np.nan); vv = np.zeros((P, 240))
    c = d["close"].to_numpy(float)
    lc[pi, m] = np.log(np.where(c > 0, c, np.nan)); vv[pi, m] = np.nan_to_num(d["volume"].to_numpy(float))
    lc = pd.DataFrame(lc).ffill(axis=1).to_numpy()
    cv = np.cumsum(vv, 1)
    rows = []
    for s0, k in SLOTK.items():
        e = k - 1
        out = {"iid": pairs.get_level_values(0), "day": pairs.get_level_values(1), "slot": s0}
        for n in NS:
            target = cv[:, e] - n * cv[:, e] / k                                   # 窗口量 = 今日平均每分钟量 × n
            cnt = (cv[:, :e + 1] <= target[:, None]).sum(1)                        # 累计量 ≤ 目标的分钟数
            j0 = np.maximum(cnt - 1, 0)
            out["VC_R%d" % n] = np.where(cv[:, e] > 0, lc[:, e] - lc[np.arange(P), j0], np.nan)
            if n == 30:
                out["VC_LEN30"] = np.where(cv[:, e] > 0, e - j0, np.nan).astype(float)
        rows.append(pd.DataFrame(out))
    return pd.concat(rows, ignore_index=True)


def main():
    with Pool(int(os.environ.get("NPROC", 6))) as p:
        parts = p.map(one, sorted(glob.glob(SRC)))
    A = pd.concat(parts, ignore_index=True)
    print("行数", len(A), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    for nm in ["VC_R10", "VC_R30", "VC_R60", "VC_LEN30"]:
        Wd = A.pivot_table(index=["iid", "day"], columns="slot", values=nm, aggfunc="last").reindex(columns=H1.CS).sort_index()
        x = H1.align(c, Wd)
        x = np.where(np.isfinite(x) & (x == 0), 1e-9, x); x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((np.nan_to_num(x) != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
