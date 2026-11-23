# -*- coding: utf-8 -*-
"""第十六轮：K1 收盘段季节性核实（v_D2CLS）+ K2 日内不变锚扫描 + K3 其他时刻的特殊段。

口径：engine.Core.evaluate；锚一律用"严格过去交易日"构造，日内各截面取值相同（当日不变）。
脚本依赖 h1_season.py 的段缓存 data/bar30m → /srv/media/jyc_scorer/vc_tmp/h1_seg.parquet。
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_optimize as O  # noqa: E402
from g_go import GO, WINS, IXW, _en  # noqa: E402
from h1_stack import build as h1build, D1B, g4low_delta  # noqa: E402
from h1_season import CACHE, build_cache, wide, roll_prev, align, CS, SEGS  # noqa: E402

OUT = pathlib.Path("/srv/media/jyc_scorer/vc_tmp")
WINMAP = [("2023", 0), ("2024", 0), ("2021", 1), ("2022", 1), ("2020", 2)]
RANK_SD = 0.2887          # 均匀分布 [-0.5,0.5] 的标准差
A_RANK = 0.05             # 秩口径：全天小权重 a=0.05（Claude M2b），14:30 再加一份 b=0.10


def load_frames():
    if not CACHE.exists():
        build_cache()
    t = pd.read_parquet(CACHE)
    t["day"] = pd.to_datetime(t["day"])
    W = wide(t)
    close, open_, amt, vol = W["close"], W["open"], W["amount"], W["volume"]
    vwap = amt / vol.where(vol > 0)
    lab = close[SEGS] / vwap[SEGS] - 1.0
    return {"lab": lab, "close": close, "open": open_, "amt": amt, "vol": vol}


def make_anchors(D):
    lab, close, amt, vol = D["lab"], D["close"], D["amt"], D["vol"]
    tot_a = amt[SEGS].sum(axis=1)
    tot_v = vol[SEGS].sum(axis=1)
    A = {}
    for N in (5, 10, 20, 60):
        A["CLS%d" % N] = roll_prev(lab["1500"], N, max(N // 2, 3))
    A["LAB1500p"] = lab["1500"].groupby(level=0, sort=False).shift(1)
    A["ALLSEG20"] = roll_prev(lab[SEGS].mean(axis=1), 20, 10)
    A["OPEN20"] = roll_prev(lab["1000"], 20, 10)
    A["AMTSH20"] = roll_prev(amt["1500"] / tot_a.where(tot_a > 0), 20, 10)
    dvwap = close["1500"] / (tot_a / tot_v.where(tot_v > 0)) - 1.0
    A["DAYVWAP20"] = roll_prev(dvwap, 20, 10)
    A["CLOSEPOS20"] = roll_prev(close["1500"] / close[SEGS].max(axis=1) - 1.0, 20, 10)
    return A


def daily_to_slots(s):
    return pd.DataFrame({s0: s for s0 in CS})


def rank_of(c, series, z930):
    x = np.nan_to_num(align(c, daily_to_slots(series)), nan=0.0)
    x = np.where(z930, 0.0, x)
    r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
    return x, r


def date_mask_of(c, start, end, slots=None):
    """per_date_rank_corr 的 date_mask 是按 date_id 索引的（长度 n_dates），不是逐行掩码。"""
    kd = pd.to_datetime(c.key_date)
    sel = (kd >= pd.Timestamp(start)) & (kd <= _en(end))
    if slots is not None:
        sel = sel & np.isin(c.slot_of_date[c.date_id], list(slots))
    m = np.zeros(c.n_dates, bool)
    m[np.unique(c.date_id[sel])] = True
    return m


def xsec_sd(c, x, st, en):
    kd = pd.to_datetime(c.key_date)
    m = (kd >= st) & (kd <= _en(en)) & np.isfinite(x) & (c.slot_of_date[c.date_id] != 930)
    if m.sum() < 100:
        return np.nan
    s = pd.Series(np.asarray(x, float)[m]).groupby(c.date_id[m]).std()
    return float(np.nanmedian(s.to_numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="k2")
    ap.add_argument("--submit", default="")
    a = ap.parse_args()
    G = GO()
    c = G.c
    kd = pd.to_datetime(c.key_date)
    z930 = c.slot_of_date[c.date_id] == 930
    slot = c.slot_of_date[c.date_id]
    D = load_frames()
    AN = make_anchors(D)
    print("锚：%s" % list(AN), flush=True)

    # 底座（原始值，逐区间）
    B137 = {ix: np.asarray(R.load(("v_D2OPT3_raw", "v_D2OPT3O21_raw", "v_D2OPT3O20_raw")[ix])[0], float)
            for ix in (0, 1, 2)}
    B1398 = {ix: h1build(G, ix, "s161", D1B) for ix in (0, 1, 2)}
    B1399 = {ix: h1build(G, ix, "1363", g4low_delta()) for ix in (0, 1, 2)}
    yr = kd.year.to_numpy()
    H2 = np.zeros(c.n_rows)
    for y in ("2020", "2021", "2022", "2023", "2024"):
        v = np.load(OUT / ("h2_seas_%s.npy" % y))
        H2[yr == int(y)] = v[yr == int(y)]
    BASES = {"1397": B137, "1398": B1398, "1399": B1399, "H2SEAS": {ix: H2 for ix in (0, 1, 2)}}

    def ev(B, wn):
        st, en = [q for q in WINS if q[0] == wn][0][1:3]
        m, _ = c.evaluate(np.where(z930, 0.0, np.asarray(B, float)), start=st, end=_en(en))
        return m

    if a.mode == "k1":
        # 权重换算：a = 0.05 * σ_底座 / σ_秩
        for bn, B in BASES.items():
            print("\n=== 底座 %s ===" % bn, flush=True)
            base_met = {wn: ev(B[ix], wn) for wn, ix in WINMAP}
            print("  底座 " + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
                wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in WINMAP),
                flush=True)
            for an in ("CLS20",):
                _, r = rank_of(c, AN[an], z930)
                for use1430 in (True, False):
                    cells = []
                    for wn, ix in WINMAP:
                        st, en = [q for q in WINS if q[0] == wn][0][1:3]
                        sd = xsec_sd(c, B[ix], st, en)
                        aa = A_RANK * sd / RANK_SD
                        bb = 2 * aa if use1430 else 0.0
                        x = np.where(np.isfinite(B[ix]), np.asarray(B[ix], float), np.nan)
                        comp = x + aa * r + bb * r * (slot == 1430)
                        m = ev(np.where(z930, 0.0, comp), wn)
                        b = base_met[wn]
                        cells.append("%s +%.5f/+%.4f/+%.1f/+%.4f/%+.4f(a=%.4f)%s" % (
                            wn, m["ic_mean"] - b["ic_mean"], m["ic_ir"] - b["ic_ir"], m["ls_sharpe"] - b["ls_sharpe"],
                            m["stress"] - b["stress"], m["turnover"] - b["turnover"], aa,
                            "★" if (m["ic_mean"] >= b["ic_mean"] and m["ic_ir"] >= b["ic_ir"]
                                    and m["stress"] >= b["stress"] and m["turnover"] <= b["turnover"] + 1e-9) else ""))
                    print("  %s%s %s" % (an, "" if use1430 else "（不加14:30）", " | ".join(cells)), flush=True)
        if a.submit:
            _, r = rank_of(c, AN["CLS20"], z930)
            Bs = {}
            for wn, ix in (("main", 0), ("o21", 1), ("o20", 2)):
                st, en = ("2023-01-01", "2024-12-31") if ix == 0 else (("2021-01-01", "2022-12-31") if ix == 1 else ("2020-01-01", "2020-12-31"))
                sd = xsec_sd(c, B137[ix], st, en)
                aa = A_RANK * sd / RANK_SD
                x = np.where(np.isfinite(B137[ix]), np.asarray(B137[ix], float), np.nan)
                Bs[wn] = np.where(z930, 0.0, x + aa * r + 2 * aa * r * (slot == 1430))
            G.write_submit(a.submit, Bs, True)

    elif a.mode == "k2":
        base = B1398
        base_met = {wn: ev(base[ix], wn) for wn, ix in WINMAP}
        print("底座 1398：" + " ".join("%s IC %.5f IR %.4f 换手 %.4f" % (
            wn, base_met[wn]["ic_mean"], base_met[wn]["ic_ir"], base_met[wn]["turnover"]) for wn, _ in WINMAP), flush=True)
        y = np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)
        ks = [(2023, "2023-01-01", "2023-12-31"), (2024, "2024-01-01", "2024-12-31")]
        rows = []
        for name, s in AN.items():
            x, r = rank_of(c, s, z930)
            ic = []
            for yy, st, en in ks:
                dm = date_mask_of(c, st, en, slots=(1000, 1030, 1100, 1130, 1330, 1400, 1430))
                ic.append(c.per_date_rank_corr(r, y, dm))
            # 单锚合成（a=0.05 秩口径）
            cells, oks = [], True
            for wn, ix in WINMAP:
                st, en = [q for q in WINS if q[0] == wn][0][1:3]
                sd = xsec_sd(c, base[ix], st, en)
                aa = A_RANK * sd / RANK_SD
                xx = np.where(np.isfinite(base[ix]), np.asarray(base[ix], float), np.nan)
                m = ev(np.where(z930, 0.0, xx + aa * r), wn)
                b = base_met[wn]
                ok = (m["ic_mean"] >= b["ic_mean"] and m["turnover"] <= b["turnover"] + 1e-9)
                oks &= ok
                cells.append("%s %+.5f/%+.4f/%+.4f" % (wn, m["ic_mean"] - b["ic_mean"],
                                                      m["ic_ir"] - b["ic_ir"], m["turnover"] - b["turnover"]))
            rows.append((name, ic[0], ic[1], oks, cells))
        for name, i23, i24, oks, cells in sorted(rows, key=lambda z: -max(abs(z[1]), abs(z[2]))):
            print("%-11s 单锚IC 23 %+.4f 24 %+.4f %s" % (name, i23, i24, "★" if oks else " "), flush=True)
            print("    ΔIC/ΔIR/Δ换手 " + " | ".join(cells), flush=True)

    elif a.mode == "slots":
        # K3：逐截面（槽位）的"下一窗口季节性"IC
        for nm in ("L_20", "L_5"):
            x = np.load(OUT / ("h1_%s.npy" % nm))
            r = np.nan_to_num(O.rc_of(x, c.date_id, c.n_dates), nan=0.0)
            y = np.nan_to_num(O.rc_of(c.y, c.date_id, c.n_dates), nan=0.0)
            print("\n=== %s 逐槽位 IC（截面口径，非 09:30）===" % nm, flush=True)
            hdr = "槽位 " + " ".join("   %s   " % s for s in ("10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30"))
            for yy, st, en in [(2023, "2023-01-01", "2023-12-31"), (2024, "2024-01-01", "2024-12-31"),
                               (2021, "2021-01-01", "2021-12-31"), (2022, "2022-01-01", "2022-12-31"),
                               (2020, "2020-01-01", "2020-12-31")]:
                cells = []
                for s in (1000, 1030, 1100, 1130, 1330, 1400, 1430):
                    dm = date_mask_of(c, st, en, slots=(s,))
                    cells.append("%+.4f" % c.per_date_rank_corr(r, y, dm))
                print("%d %s" % (yy, " ".join("%8s" % v for v in cells)), flush=True)
            if yy == 2020:
                print(hdr, flush=True)


if __name__ == "__main__":
    main()
