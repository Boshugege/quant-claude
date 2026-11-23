# -*- coding: utf-8 -*-
"""新旧两版因子在"双方都有完整 5 日历史"的区段内逐值对比。
用法：python eq_tail.py <新文件> <旧文件> <start> <end> <只比较该日期起>
旧文件（1323/1273）没有 warmup，其余区段会少历史；本脚本把比较限制在 warmup 之后。"""
import importlib.util, sys
sys.path.insert(0, "/home/user/jyc_scorer/platform_sim")
import numpy as np
import pandas as pd
ds = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}


def run(f, s, e):
    spec = importlib.util.spec_from_file_location("m", f)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m.main(ds, s, e)


a = run(sys.argv[1], sys.argv[3], sys.argv[4])
b = run(sys.argv[2], sys.argv[3], sys.argv[4])
c = a.merge(b, on=["date", "instrument"], how="outer", suffixes=("_new", "_old"))
c["date"] = pd.to_datetime(c["date"])
c = c[c["date"] >= pd.Timestamp(sys.argv[5])]
d = (c["factor_new"] - c["factor_old"]).abs()
print(f"[{sys.argv[5]} 起] 行数 {len(c):,}；最大差 {d.max():.3e}；平均差 {d.mean():.3e}；"
      f"差>1e-6 的行 {(d > 1e-6).sum()}；相关 {c['factor_new'].corr(c['factor_old']):.8f}")
