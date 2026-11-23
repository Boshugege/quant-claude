# -*- coding: utf-8 -*-
"""新旧两版因子文件在同一区间的输出逐值对比（评测机本机，platform_sim）。用法：python eq_versions.py <新文件> <旧文件> <start> <end>"""
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
d = (c["factor_new"] - c["factor_old"]).abs()
print(f"行数 新 {len(a):,} 旧 {len(b):,} 合并 {len(c):,}；最大差 {d.max():.3e}；差 > 1e-9 的行 {(d > 1e-9).sum()}；相关 {c['factor_new'].corr(c['factor_old']):.6f}")
