# -*- coding: utf-8 -*-
"""E2（10-19）：价格同步性（Morck 等的价格信息含量）。个股 30 分钟段收益对市场段收益（截面缩尾均值）在过去 20 个交易日、
每天 8 段上的 R²（相关系数平方），只用前几日、当日各截面相同。另存同窗的 β（看风格剔除后还剩多少）。
保存 vc_tmp/h1_SYNC20.npy、h1_BETA20.npy（原始值），并打印单独五窗口指标（两个方向）与相对 1610 的叠加扫描。"""
import numpy as np
import pandas as pd
import root_eval as R
import h1_season as H1


def main():
    if not H1.CACHE.exists():
        H1.build_cache()
    t = pd.read_parquet(H1.CACHE)
    t["day"] = pd.to_datetime(t["day"])
    W = H1.wide(t)
    del t
    close, open_ = W["close"], W["open"]
    S = H1.SEGS
    prev = close[S].shift(1, axis=1)
    prev["1000"] = open_["1000"]
    y = (close[S] / prev - 1.0).replace([np.inf, -np.inf], np.nan)          # 个股段收益 (iid, day) × 8 段
    day = y.index.get_level_values(1)
    mk = {}
    for s0 in S:
        g = pd.DataFrame({"day": day, "v": y[s0].to_numpy(float)})
        q = g.groupby("day")["v"].quantile([0.01, 0.99]).unstack()
        g = g.join(q, on="day")
        mk[s0] = g.assign(v=g["v"].clip(g[0.01], g[0.99])).groupby("day")["v"].mean()
    M = pd.DataFrame(mk)                                                  # day × 段
    x = pd.DataFrame(M.reindex(day).to_numpy(), index=y.index, columns=S)   # 对齐到个股行
    ok = y.notna() & x.notna()
    yv, xv = y.where(ok), x.where(ok)
    agg = pd.DataFrame({"n": ok.sum(1), "sx": xv.sum(1), "sy": yv.sum(1), "sxx": (xv * xv).sum(1),
                        "syy": (yv * yv).sum(1), "sxy": (xv * yv).sum(1)})
    roll = agg.groupby(level=0, sort=False).rolling(20, min_periods=10).sum()
    roll.index = roll.index.droplevel(0)
    roll = roll.reindex(agg.index).groupby(level=0, sort=False).shift(1)     # 严格只用前几日
    n = roll["n"].where(roll["n"] >= 80)
    cxy = roll["sxy"] - roll["sx"] * roll["sy"] / n
    cxx = roll["sxx"] - roll["sx"] ** 2 / n
    cyy = roll["syy"] - roll["sy"] ** 2 / n
    r2 = (cxy ** 2 / (cxx * cyy)).where((cxx > 0) & (cyy > 0))
    beta = (cxy / cxx).where(cxx > 0)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    for nm, ser in (("SYNC20", r2), ("BETA20", beta)):
        B = pd.DataFrame({s0: ser.to_numpy() for s0 in H1.CS}, index=y.index)
        v = H1.align(c, B)
        v[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(v, nan=0.0))
        print("保存", nm, "非零率 %.3f 均值 %.4f" % (float((v != 0).mean()), float(np.nanmean(np.where(v != 0, v, np.nan)))), flush=True)


if __name__ == "__main__":
    main()
