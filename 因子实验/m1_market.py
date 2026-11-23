# -*- coding: utf-8 -*-
"""F11 截面状态特征（10-19）：每个截面一个值、对全部股票相同，只能被树当作"开关"用（按市况调整各分量的权重），不改变单独的排序。
全部只用截面之前的数据（30 分钟 bar，段 (t−30, t] 在 t 时已知）：
  MKT_SEG   最近一段收益的截面均值（1%／99% 缩尾）
  MKT_DAY   今日开盘至今收益的截面均值（缩尾）
  MKT_DISP  最近一段收益的截面标准差（缩尾后）
  MKT_PACE  今日累计成交额 ÷ 过去 20 日同一时刻累计成交额均值的截面中位数
保存为 vc_tmp/h1_MKT_*.npy（原始值，h2_robust 读入时不排名）。"""
import numpy as np
import pandas as pd
import root_eval as R
import h1_season as H1


def wins(df, q=0.01):
    lo = df.quantile(q, axis=1)
    hi = df.quantile(1 - q, axis=1)
    return df.clip(lo, hi, axis=0)


def main():
    if not H1.CACHE.exists():
        H1.build_cache()
    t = pd.read_parquet(H1.CACHE)
    t["day"] = pd.to_datetime(t["day"])
    W = H1.wide(t)
    del t
    close, open_, amt = W["close"], W["open"], W["amount"]
    CS = H1.CS
    prev = close[H1.SEGS].shift(1, axis=1)
    prev["1000"] = open_["1000"]
    seg = (close[CS] / prev[CS] - 1.0).replace([np.inf, -np.inf], np.nan)
    dayr = close[CS].div(open_["1000"].where(open_["1000"] > 0), axis=0) - 1.0
    cum = amt[H1.SEGS].cumsum(axis=1)[CS]
    prevcum = H1.roll_prev(cum, 20, 10)
    pace = (cum / prevcum.where(prevcum > 0)).replace([np.inf, -np.inf], np.nan)
    day = seg.index.get_level_values(1)
    out = {}
    for nm, fr, how in (("MKT_SEG", seg, "mean"), ("MKT_DAY", dayr, "mean"), ("MKT_DISP", seg, "std"), ("MKT_PACE", pace, "median")):
        per = {}
        for s0 in CS:
            col = fr[s0]
            g = pd.DataFrame({"day": day, "v": col.to_numpy(float)})
            if how in ("mean", "std"):
                q = g.groupby("day")["v"].quantile([0.01, 0.99]).unstack()
                g = g.join(q, on="day")
                g["v"] = g["v"].clip(g[0.01], g[0.99])
            per[s0] = getattr(g.groupby("day")["v"], how)()
        M = pd.DataFrame(per)                                             # 索引 day，列 slot
        B = pd.DataFrame(index=fr.index, columns=CS, dtype=float)          # 广播回 (iid, day)
        for s0 in CS:
            B[s0] = M[s0].reindex(day).to_numpy()
        out[nm] = B
        print(nm, M.describe().loc[["mean", "std"]].round(5).to_dict(), flush=True)
    # F11b：日内不变的市况（只用前几日的日收益，当日各截面相同）
    c15 = close["1500"]
    dret = (c15 / c15.groupby(level=0).shift(1) - 1.0).replace([np.inf, -np.inf], np.nan)
    g = pd.DataFrame({"day": day, "v": dret.to_numpy(float)})
    q = g.groupby("day")["v"].quantile([0.01, 0.99]).unstack()
    g = g.join(q, on="day")
    g["v"] = g["v"].clip(g[0.01], g[0.99])
    mret = g.groupby("day")["v"].mean().sort_index()
    mdisp = g.groupby("day")["v"].std().sort_index()
    dd = {"MKD_RET1": mret.shift(1), "MKD_RET5": mret.rolling(5, min_periods=3).sum().shift(1),
          "MKD_VOL20": mret.rolling(20, min_periods=10).std().shift(1), "MKD_DISP5": mdisp.rolling(5, min_periods=3).mean().shift(1)}
    for nm, ser in dd.items():
        B = pd.DataFrame(index=seg.index, columns=CS, dtype=float)
        v = ser.reindex(day).to_numpy()
        for s0 in CS:
            B[s0] = v
        out[nm] = B
        print(nm, "均值 %.5f 标准差 %.5f" % (ser.mean(), ser.std()), flush=True)
    c = R.core()
    z930 = c.slot_of_date[c.date_id] == 930
    for nm, B in out.items():
        x = H1.align(c, B)
        x[z930] = 0.0
        np.save(H1.OUT / ("h1_%s.npy" % nm), np.nan_to_num(x, nan=0.0))
        print("保存", nm, "非零率 %.3f" % float((x != 0).mean()), flush=True)


if __name__ == "__main__":
    main()
