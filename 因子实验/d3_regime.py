# -*- coding: utf-8 -*-
"""D3 状态条件化：按"前一交易日"的市场状态切换权重，验证能不能救 2020。

状态量（全部用前一交易日及更早，来自本地暴露表的日收益）：
  disp  截面收益离散度（std）
  vol5  市场指数收益的 5 日滚动波动
  mkt   前一交易日市场（权重加权）收益
分组：用**过去 250 个交易日的中位数**做高／低切分（严格因果）。
合成：B = 1363原始值 + w(状态)·rank(155)。在 2020-01～2022-12（拟合）上搜 (w_hi, w_lo)，
再看 2020 单独、2023／2024 主区间。
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import root_eval as R  # noqa: E402
import d2_eval as D  # noqa: E402
import d2_optimize as O  # noqa: E402

STATE = pathlib.Path("/home/user/jyc_scorer/data/exposure/_daily_state.parquet")


def state_table():
    s = pd.read_parquet(STATE).sort_index()
    s["vol5"] = s["mkt"].rolling(5, min_periods=3).std()
    s["mkt_prev"] = s["mkt"]                # 行=当日（收盘后可知），下面统一再 shift 1 个交易日
    s = s.shift(1)
    return s


def regime_flag(s, kind):
    """返回 布尔 Series（True = 高状态 / 上涨），阈值用过去 250 日中位数（因果）。"""
    col = {"disp": "disp", "vol5": "vol5", "mkt": "mkt_prev"}[kind]
    v = s[col]
    thr = v.rolling(250, min_periods=60).median()
    if kind == "mkt":
        return (v > 0).fillna(False)
    return (v > thr).fillna(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ws", default="0.0,0.1,0.2,0.35,0.5")
    a = ap.parse_args()
    c = R.core()
    nd = c.n_dates
    grid = [float(x) for x in a.ws.split(",")]

    H = R.load("v_S1284z1342_raw_wm0100")[0]                 # 主区间 1363
    r155 = O.axis(c, "fm_s155", "2023-01-01", "2024-12-31")  # 主区间 rank(s155)
    Hfit = np.where(np.isfinite(R.load("v_V1363O21_raw_wm0100")[0]),
                    R.load("v_V1363O21_raw_wm0100")[0],
                    R.load("v_V1363O20_raw_wm0100")[0])      # 拟合窗 1363
    r155fit = np.where(np.isfinite(r155), r155, np.nan)
    r21 = O.axis(c, "o21_s155", "2021-01-01", "2022-12-31")
    r20 = O.axis(c, "o20_s155", "2020-01-01", "2020-12-31")
    r155fit = np.where(np.isfinite(r21), r21, r20)

    # 行 → 日历日 → 状态
    days = pd.to_datetime([pd.Timestamp(int(x)) for x in c.dates]).normalize()
    s = state_table()
    def rows_flag(kind):
        f = regime_flag(s, kind)
        v = np.array([bool(f.get(d, False)) for d in days])
        return v[c.date_id]

    def build(Hrow, rrow, flagrow, w_hi, w_lo):
        w = np.where(flagrow, w_hi, w_lo)
        return Hrow + w * np.nan_to_num(rrow, nan=0.0)

    def ev(B, st, en):
        m, _s = c.evaluate(B, start=st, end=en)
        return m

    def J(m):
        return m["ic_ir"] + m["stress"] - 0.2 * m["turnover"]

    FIT = ("2020-01-01", "2022-12-31")
    m0 = ev(Hfit, *FIT)
    print("拟合窗 1363（w≡0）：IC %.5f IR %.4f 压力 %.4f 换手 %.4f J %.4f"
          % (m0["ic_mean"], m0["ic_ir"], m0["stress"], m0["turnover"], J(m0)))
    flat = ev(Hfit + 0.2 * np.nan_to_num(r155fit, nan=0.0), *FIT)
    print("拟合窗 1363+155×0.20（固定）：IC %.5f IR %.4f 压力 %.4f 换手 %.4f J %.4f"
          % (flat["ic_mean"], flat["ic_ir"], flat["stress"], flat["turnover"], J(flat)))

    results = []
    for kind in ("disp", "vol5", "mkt"):
        fl = rows_flag(kind)
        best = None
        for wh in grid:
            for wl in grid:
                B = build(Hfit, r155fit, fl, wh, wl)
                m = ev(B, *FIT)
                if best is None or J(m) > best[0]:
                    best = (J(m), wh, wl, m)
        results.append((kind, best))
        # 主区间（分段验证：2023、2024、2020 单独）
        flv = rows_flag(kind)
        Bv = build(H, r155, flv, best[1], best[2])
        mv = ev(Bv, "2023-01-01", "2024-12-31")
        m23 = ev(Bv, "2023-01-01", "2023-12-31")
        m24 = ev(Bv, "2024-01-01", "2024-12-31")
        m20 = ev(build(np.where(np.isfinite(R.load("v_V1363O20_raw_wm0100")[0]),
                               R.load("v_V1363O20_raw_wm0100")[0], np.nan),
                       r20, flv, best[1], best[2]), "2020-01-01", "2020-12-31")
        print("\n== %s == 最好 (w_hi %.2f, w_lo %.2f)：拟合 J %.4f（IC %.5f IR %.4f 压力 %.4f 换手 %.4f）"
              % (kind, best[1], best[2], best[0], best[3]["ic_mean"], best[3]["ic_ir"], best[3]["stress"], best[3]["turnover"]))
        base20 = ev(np.where(np.isfinite(R.load("v_V1363O20_raw_wm0100")[0]),
                             R.load("v_V1363O20_raw_wm0100")[0], np.nan), "2020-01-01", "2020-12-31")
        print("   2020 单独：IC %+.5f IR %+.4f 压力 %+.4f 换手 %+.4f（相对 1363：IC %+.5f IR %+.4f 压力 %+.4f 换手 %+.4f）"
              % (m20["ic_mean"], m20["ic_ir"], m20["stress"], m20["turnover"],
                 m20["ic_mean"] - base20["ic_mean"], m20["ic_ir"] - base20["ic_ir"],
                 m20["stress"] - base20["stress"], m20["turnover"] - base20["turnover"]))
        print("   2023-24 ：IC %+.5f IR %+.4f 压力 %+.4f 换手 %+.4f" % (mv["ic_mean"], mv["ic_ir"], mv["stress"], mv["turnover"]))
        print("   2023    ：IC %+.5f IR %+.4f 换手 %+.4f ; 2024：IC %+.5f IR %+.4f 换手 %+.4f"
              % (m23["ic_mean"], m23["ic_ir"], m23["turnover"], m24["ic_mean"], m24["ic_ir"], m24["turnover"]))


if __name__ == "__main__":
    main()
