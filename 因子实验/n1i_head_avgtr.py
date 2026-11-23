# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）补充：头部自己的 c_avgtr 依赖影响多大？

头部信号 = rank_blend([ladder, c_upsh, c_avgtr, r_rv16, c_body], [1.0, 0.30, 0.15, 0.20, 0.15])，
其中 c_avgtr = Σ成交额 ÷ Σ成交笔数（用 deal_number）。若公榜 deal_number 为空，这一列整列 NaN，
rank_blend 会按可用分量重新归一 → 头部变成"四分量再归一"。

本脚本把 1363 里的 _head_impl 抄两份（原版 / c_avgtr 强制 NaN），在同一区间跑，比较：
  * 因子秩相关；* 逐截面 IC／IC_IR／换手（用 engine.Core 的本地标签）。
用法：python 因子实验/n1i_head_avgtr.py [--q 2024Q4]
"""
import argparse
import pathlib
import sys
import types

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/user/jyc_scorer/platform_sim")

import d2_optimize as O          # noqa: E402
import root_eval as R            # noqa: E402
from g_go import GO, inst_codes  # noqa: E402

SRC = pathlib.Path("/home/user/quant-claude/因子代码/1363_合成头部加九根含六十九十分钟资金流精简按月头部按季.py")
DS = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments",
      "exposure": "cpt_jyc_2026_exposure"}


def head_module(zero_avgtr):
    txt = SRC.read_text(encoding="utf-8")
    if zero_avgtr:
        old = '_avgtr = df["sum_amt"] / df["sum_deals"].where(df["sum_deals"] > 0)'
        assert old in txt
        txt = txt.replace(old, '_avgtr = pd.Series(np.nan, index=df.index)')
    mod = types.ModuleType("head_zd" if zero_avgtr else "head_raw")
    exec(compile(txt, str(SRC), "exec"), mod.__dict__)
    return mod


def axis(c, df):
    f = pd.DataFrame({"row": np.arange(c.n_rows, dtype=np.int64), "date": pd.to_datetime(c.key_date),
                      "instrument": inst_codes(c)})
    d = df.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["instrument"] = d["instrument"].astype(str)
    m = f.merge(d, on=["date", "instrument"], how="left", sort=False)
    v = np.full(c.n_rows, np.nan)
    v[m["row"].to_numpy()] = m["factor"].to_numpy(float)
    return np.nan_to_num(O.rc_of(v, c.date_id, c.n_dates), nan=0.0), v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-10-01 00:00:00")
    ap.add_argument("--end", default="2024-12-31 23:59:59")
    a = ap.parse_args()
    G = GO()
    c = G.c
    out = {}
    for zd in (False, True):
        mod = head_module(zd)
        df = mod._head_impl(DS, a.start, a.end)
        ax, raw = axis(c, df)
        out["zd" if zd else "raw"] = (ax, raw, df)
        print("跑完 c_avgtr=%s：%d 行" % ("NaN" if zd else "真值", len(df)), flush=True)

    axr, rawr, dfr = out["raw"]
    axz, rawz, dfz = out["zd"]
    m = pd.DataFrame({"d": pd.to_datetime(c.key_date), "i": inst_codes(c), "r": rawr, "z": rawz})
    m = m[m["r"].notna() & m["z"].notna()]
    rr = m.groupby("d").apply(lambda g: g["r"].corr(g["z"], method="spearman")).mean()
    print("\n头部（2024Q4）：c_avgtr 真值 vs 强制 NaN 的逐截面秩相关 = %.4f" % rr)

    st, en = "2024-10-01", "2024-12-31"
    en_ts = pd.Timestamp(en).normalize() + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    for nm, raw in (("原版(含 c_avgtr)", rawr), ("c_avgtr=NaN(平台等效)", rawz)):
        v = np.where(G.z930, 0.0, raw)
        mm, _ = c.evaluate(np.where(np.isfinite(raw), v, np.nan), start=st, end=en_ts)
        print("  %-24s IC %.5f  IC_IR %.4f  多空 %6.2f  压力 %.4f  换手 %.4f"
              % (nm, mm["ic_mean"], mm["ic_ir"], mm["ls_sharpe"], mm["stress"], mm["turnover"]))


if __name__ == "__main__":
    main()
