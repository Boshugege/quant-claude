# -*- coding: utf-8 -*-
"""第十六轮 K2：把"日内不变锚"做成方法。

思路（K1 的副产品）：一个**当日不变**（只用严格过去交易日数据）、但**单独有 IC** 的分量加进合成后，
相邻截面的排序被"钉住"，换手反而下降。于是把它拆成两个可筛的维度：
  - 单锚 IC      → 能不能提分；
  - 日间持续性   → 相邻交易日截面秩相关的均值，越高越能"钉住"排序、越能降换手。
阶段 A 廉价筛全部锚（只算秩相关，不做 evaluate），阶段 B 只对头部锚做完整 evaluate。

口径：engine.Core.evaluate；锚一律严格过去交易日构造、日内各截面同值、09:30 置 0。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import d2_optimize as O  # noqa: E402
import k1_anchor as K  # noqa: E402
from g_go import GO, WINS, _en  # noqa: E402
from h1_stack import build as h1build, D1B, g4low_delta  # noqa: E402
from h1_season import SEGS, roll_prev  # noqa: E402


def make_anchors2(D):
    """在 K1 的 10 个锚之外，补 4 个"当日不变"的类日频锚。"""
    A = K.make_anchors(D)
    close, amt, vol = D["close"], D["amt"], D["vol"]
    c1 = close["1500"]
    g = c1.groupby(level=0, sort=False)
    dret = g.pct_change()
    A["MOM20"] = roll_prev(dret, 20, 15, how="sum")      # 过去 20 日累计收益
    A["MOM60"] = roll_prev(dret, 60, 45, how="sum")
    A["VOL20"] = roll_prev(dret.abs(), 20, 15)           # 20 日均绝对日收益
    dayamt = amt[SEGS].sum(axis=1)
    A["AMTLN20"] = roll_prev(np.log(dayamt.where(dayamt > 0)), 20, 10)   # 20 日均成交额（对数）
    return A


def persistence(c, x, slot_of_date, slot=1000):
    """日间持续性：相邻交易日、同一截面的截面秩（Spearman）相关的均值。"""
    xv = np.asarray(x, float).copy()
    xv[slot_of_date[c.date_id] != slot] = np.nan
    m = np.isfinite(xv)
    idx = np.flatnonzero(m)
    if idx.size < 1000:
        return np.nan
    r = O.rc_of(xv, c.date_id, c.n_dates)[idx]
    df = pd.DataFrame({"d": c.date_id[idx], "i": c.inst[idx], "r": r})
    w = df.pivot_table(index="d", columns="i", values="r", aggfunc="first")
    return float(w.corrwith(w.shift(1), axis=1).mean())


def mutual_corr(AN, names, c, slot_of_date, slot=1000):
    """锚之间的相似度（用单截面的日值秩做池化相关，仅用于去重挑选）。"""
    cols = {}
    m = slot_of_date[c.date_id] == slot
    for n in names:
        cols[n] = K.align(c, K.daily_to_slots(AN[n])).copy()
    X = pd.DataFrame({n: v[m] for n, v in cols.items()})
    return X.corr(method="spearman")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="both")
    ap.add_argument("--bases", default="1398,1399")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--pool", type=int, default=5)
    ap.add_argument("--rd", type=float, default=0.7, help="去重阈值：与已选锚 |corr| 高于此值则跳过")
    a = ap.parse_args()
    G = GO()
    c = G.c
    kd = pd.to_datetime(c.key_date)
    z930 = c.slot_of_date[c.date_id] == 930
    slot = c.slot_of_date[c.date_id]
    D = K.load_frames()
    AN = make_anchors2(D)
    print("锚（%d 个）：%s" % (len(AN), list(AN)), flush=True)
    y = np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)
    RS = {}

    # ---------- 阶段 A：廉价筛选 ----------
    rows = []
    for name, s in AN.items():
        x, r = K.rank_of(c, s, z930)
        RS[name] = (x, r)
        ics = {}
        for wn, ix in K.WINMAP:
            st, en = [q for q in WINS if q[0] == wn][0][1:3]
            ics[wn] = c.per_date_rank_corr(r, y, K.date_mask_of(c, st, en))
        va = [ics[wn] for wn, _ in K.WINMAP]
        rows.append((name, va, float(np.nanmean(va)), persistence(c, x, c.slot_of_date)))
    rows.sort(key=lambda z: -z[2])
    print("\n=== 阶段 A：单锚 IC 与日间持续性（IC 按 |均值| 排序）===", flush=True)
    print("%-11s %s %8s %8s" % ("锚", " ".join("%9s" % w for w, _ in K.WINMAP), "IC均值", "持续性"), flush=True)
    for name, va, mu, per in rows:
        print("%-11s %s %+8.5f %8.3f" % (name, " ".join("%+9.5f" % v for v in va), mu, per), flush=True)

    if a.stage == "a":
        return

    # ---------- 阶段 B：完整 evaluate ----------
    cw = mutual_corr(AN, [r[0] for r in rows], c, c.slot_of_date)
    picked, pool = [], []
    for name, va, mu, per in rows:
        if mu <= 0:
            continue
        if len(pool) < a.pool:
            pool.append(name)
        if len(picked) >= a.topk:
            continue
        if all(abs(cw.loc[name, p]) < a.rd for p in picked):
            picked.append(name)
    print("\n去重后单锚：%s；叠加池(%d)：%s" % (picked, len(pool), pool), flush=True)

    BASES = {}
    if "1398" in a.bases:
        BASES["1398"] = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    if "1399" in a.bases:
        BASES["1399"] = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(B, float)), start=st, end=_en(en))
        return m

    out = {}
    for bn, B in BASES.items():
        base_met = {wn: ev(B[ix], wn) for wn, ix in K.WINMAP}
        print("\n=== 底座 %s ===" % bn, flush=True)
        print("  底座 " + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
            wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in K.WINMAP), flush=True)
        cfgs = [("单锚", nm, [nm], 1.0, 100.0) for nm in picked]
        for tot in (0.05, 0.10):
            cfgs.append(("叠加%d" % (tot * 100), "+".join(pool), pool, tot, 100.0))
        for tag, nm, names, tot, _r in cfgs:
            cells, ok_all = [], True
            for wn, ix in K.WINMAP:
                st, en = [q for q in WINS if q[0] == wn][0][1:3]
                sd = K.xsec_sd(c, B[ix], st, en)
                aa = K.A_RANK * tot * sd / K.RANK_SD / len(names)
                comp = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan)
                for n2 in names:
                    comp = comp + aa * RS[n2][1]
                m = ev(np.where(z930, 0.0, comp), wn)
                b = base_met[wn]
                ok = (m["ic_mean"] >= b["ic_mean"] and m["ic_ir"] >= b["ic_ir"]
                      and m["stress"] >= b["stress"] and m["turnover"] <= b["turnover"] + 1e-9)
                ok_all &= ok
                cells.append("%s %+.5f/%+.4f/%+.4f%s" % (
                    wn, m["ic_mean"] - b["ic_mean"], m["ic_ir"] - b["ic_ir"],
                    m["turnover"] - b["turnover"], "★" if ok else ""))
            print("  [%s] %-24s %s" % (tag, nm, " ".join(cells)), flush=True)
            out[(bn, tag, nm)] = ok_all
    print("\n小结（★=五窗口 IC/IR/压力不变差且换手不升）：", flush=True)
    for k2, v in out.items():
        print("  %s %s %s -> %s" % (k2[0], k2[1], k2[2], "★" if v else "×"), flush=True)


if __name__ == "__main__":
    main()
