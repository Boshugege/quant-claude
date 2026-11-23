# -*- coding: utf-8 -*-
"""新方向对底座的增量（10-21）：c102 显示合成信号加到 1610 上 IC 不变、只换换手，所以要直接量"剔除底座之后还剩多少"。
对每个五年任务：逐截面 z 分数；与底座的相关 ρ；残差 = 信号 − ρ·底座；残差与标签的秩 IC（原始标签，未剔除风格，只作相对比较）。
按年平均，按五年最小的残差 IC 排序。09:30 不计。
用法：python c103_resid.py <底座任务1+底座任务2> <前缀> [<前缀> ...]"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

JOBS = os.path.expanduser("~/jyc_scorer/jobs")
LAB = os.path.expanduser("~/jyc_scorer/data/labels_y_off.parquet")


def jobs_by_prefix(prefixes, five_years=True):
    out = {}
    for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
        try:
            s = json.load(open(sp, encoding="utf-8"))
        except Exception:
            continue
        n, d = s.get("name") or "", os.path.dirname(sp)
        if any(n.startswith(p) for p in prefixes) and (s.get("start", "")[:4] == "2020" or not five_years) and os.path.exists(os.path.join(d, "factor_keys.parquet")):
            if n not in out or d > out[n]:
                out[n] = d
    return out


def jobdir(name):
    j = jobs_by_prefix([name], five_years=False)
    return j[name]


def keys(dirs):
    parts = [pd.read_parquet(os.path.join(d, "factor_keys.parquet")) for d in dirs]
    k = pd.concat(parts, ignore_index=True).drop_duplicates(["key_date", "instrument_id"], keep="last")
    return k.set_index(["key_date", "instrument_id"])["factor"].astype(float)


lab = pd.read_parquet(LAB)
lab = lab[lab["date"].dt.hour * 100 + lab["date"].dt.minute != 930]
lab = lab.assign(instrument_id=lab["instrument"].str[2:].astype(np.int32)).set_index(["date", "instrument_id"])["y"]
lab.index.names = ["key_date", "instrument_id"]
base_names = sys.argv[1].split("+")
B = keys([jobdir(n) for n in base_names]).reindex(lab.index)
ok = B.notna() & lab.notna()
B, y = B[ok], lab[ok]
dates = B.index.get_level_values(0)
yr = pd.Series(dates.year, index=B.index)


def zc(x):
    g = x.groupby(dates)
    return ((x - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)


def ric(x):
    """逐截面秩相关的按年均值。"""
    rx, ry = x.groupby(dates).rank(pct=True), y.groupby(dates).rank(pct=True)
    d = pd.DataFrame({"a": rx.values, "b": ry.values, "t": dates})
    c = d.groupby("t").apply(lambda q: q["a"].corr(q["b"]))
    return c.groupby(c.index.year).mean()


zB = zc(B)
icb = ric(zB)
print("底座", base_names, " ".join("%d:%.4f" % (k, v) for k, v in icb.items()), flush=True)
rows = []
for n, d in sorted(jobs_by_prefix(sys.argv[2:]).items()):
    s = pd.read_parquet(os.path.join(d, "factor_keys.parquet")).drop_duplicates(["key_date", "instrument_id"], keep="last").set_index(["key_date", "instrument_id"])["factor"].astype(float)
    s = s.reindex(B.index).fillna(0.0)
    zs = zc(s)
    rho = (zs * zB).groupby(dates).transform("mean")                       # 逐截面相关（z 分数的协方差）
    ic = ric(zs)
    sg = 1.0 if ic.mean() >= 0 else -1.0
    res = ric(sg * (zs - rho * zB))
    rr = float((zs * zB).groupby(dates).mean().mean())
    rows.append((res.min(), n, sg, rr, ic, res))
    print("%-28s 方向%+d 与底座相关 %+.2f | IC %s | 残差IC %s" % (n, sg, sg * rr, " ".join("%.4f" % (sg * v) for v in ic.values), " ".join("%.4f" % v for v in res.values)), flush=True)
print("\n== 按五年最小残差 IC 排序")
for mn, n, sg, rr, ic, res in sorted(rows, reverse=True):
    print("%-28s 最小残差IC %.4f 平均 %.4f 与底座相关 %+.2f 原IC均值 %.4f" % (n, mn, res.mean(), sg * rr, sg * ic.mean()))
