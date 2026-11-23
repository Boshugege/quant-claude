# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）：deal_number 依赖的现场验证。

两件事：
 ① 新的近 5 日根（只用收益+成交量）与原根（用单笔成交额）是**不同的信号**（截面秩相关 < 1）；
 ② 把 SQL 里的 deal_number 强制成 0（模拟公榜数据），原根立刻退化成全 0（= 公榜现象），
    新根**一点也不变**（因为它们根本没用这一列）。

用法：python 因子实验/n1e_verify.py [--start 2024-01-01] [--end 2024-12-31]
"""
import argparse
import pathlib
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/jyc_scorer/platform_sim")
FD = pathlib.Path(__file__).resolve().parent.parent / "因子代码"
DS = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
COLMAP = {"n5d": "1750", "vsd5": "1751", "q5d": "1754", "z5": "1755", "mfr5d": "1756", "net5d": "1757"}
FILES = {"1750": "1750_近五日量价同向.py", "1751": "1751_近五日分钟量离散度.py",
         "1754": "1754_近五日带符号路径效率.py", "1755": "1755_近五日零收益分钟占比.py",
         "1756": "1756_近五日上涨分钟成交量占比.py", "1757": "1757_近五日净流向占比.py"}
NEW = {"vd5": "1750", "vsd5": "1751", "q5d": "1754", "z5": "1755", "mfr5d": "1756", "net5d": "1757"}


def run(path, start, end, zero_deals=False):
    txt = path.read_text(encoding="utf-8")
    if zero_deals:
        old = "CAST(COALESCE(b.deal_number, 0) AS DOUBLE) AS n"
        assert old in txt
        txt = txt.replace(old, "CAST(0 AS DOUBLE) AS n")
    mod = types.ModuleType("m_" + path.stem + ("_zd" if zero_deals else ""))
    mod.__file__ = str(path)
    exec(compile(txt, str(path), "exec"), mod.__dict__)
    df = mod.main(DS, start, end)
    df["date"] = pd.to_datetime(df["date"])
    df["instrument"] = df["instrument"].astype(str)
    return df.rename(columns={"factor": path.stem})


def wide(df, col):
    return df.pivot_table(index="date", columns="instrument", values=col, aggfunc="last")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-06-01 00:00:00")
    ap.add_argument("--end", default="2024-12-31 23:59:59")
    a = ap.parse_args()

    print("== 跑新的去 deal_number 近 5 日根 ==")
    got = {}
    for k, fn in FILES.items():
        d = run(FD / fn, a.start, a.end)
        got[k] = d
        v = d.iloc[:, 2].to_numpy(float)
        nz = np.abs(v) > 0.5 / 4000
        print("  %-26s rows=%7d  非零占比 %.3f" % (fn, len(d), nz.mean()))

    print("\n== 新根之间是否冗余（逐日截面秩相关均值）==")
    ks = list(FILES)
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            m = got[ks[i]].merge(got[ks[j]], on=["date", "instrument"], suffixes=("_a", "_b"))
            r = m.groupby("date").apply(lambda g: g.iloc[:, 2].corr(g.iloc[:, 3], method="spearman")).mean()
            print("  %-6s vs %-6s  rho %+.3f" % (ks[i], ks[j], r))

    print("\n== 机制验证：把 deal_number 强制为 0 ==")
    old = FD / "1363_合成头部加九根含六十九十分钟资金流精简按月头部按季.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("m1363", old)
    m1363 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m1363)

    def old_root(stat, zero_deals):
        txt = old.read_text(encoding="utf-8")
        if zero_deals:
            txt = txt.replace("CAST(COALESCE(b.deal_number, 0) AS DOUBLE) AS n", "CAST(0 AS DOUBLE) AS n")
        mod = types.ModuleType("m1363z" if zero_deals else "m1363r")
        exec(compile(txt, str(old), "exec"), mod.__dict__)
        d = mod._root_impl(DS, a.start, a.end, stat, {})
        d["date"] = pd.to_datetime(d["date"])
        d["instrument"] = d["instrument"].astype(str)
        return d

    for stat in ("Y_sd5", "X_n5d", "N_dir", "W_vdir", "Z_mfr60"):
        for zd in (False, True):
            d = old_root(stat, zd)
            v = d["factor"].to_numpy(float)
            print("  %-8s deal_number%s  非零占比 %.3f" % (stat, "=0" if zd else "真值", (np.abs(v) > 0.5 / 4000).mean()))

    print("\n== 新根在 deal_number=0 下是否完全不变 ==")
    for k, fn in FILES.items():
        d1 = run(FD / fn, a.start, a.end)
        d2 = run(FD / fn, a.start, a.end, zero_deals=True)
        m = d1.merge(d2, on=["date", "instrument"], suffixes=("_1", "_2"))
        same = np.allclose(m.iloc[:, 2], m.iloc[:, 3], equal_nan=True)
        print("  %-26s 完全一致 %s（最大差 %.3g）" % (fn, same, np.nanmax(np.abs(m.iloc[:, 2] - m.iloc[:, 3]))))


if __name__ == "__main__":
    main()
