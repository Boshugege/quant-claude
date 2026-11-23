# -*- coding: utf-8 -*-
"""第十八轮 N1（更正版）：新近 5 日根与 1363 已有分量的截面秩相关矩阵。

目的：1363 的 −0.10 槽位已经有 W_vdir（当日量价同向）、−0.15/−0.10 槽位已有 Z_mfr60/Z_mfr90；
判断 Y_vd5（近 5 日量价同向）、Z_mfr5d（近 5 日上涨量占比）这类"某分量的 5 日版"到底有多冗余。
复用 1363 的精简框架（一次 SQL，逐个 STAT 走 pandas）。
"""
import pathlib
import re
import sys
import types

import numpy as np
import pandas as pd

sys.path.insert(0, "/home/user/jyc_scorer/platform_sim")
FD = pathlib.Path("/home/user/quant-claude/因子代码")
SRC = FD / "1750_近五日量价同向.py"   # 扩展版精简框架（SQL 里已含 vd5/sdv5/z5）
DS = {"bar1m": "cpt_jyc_2026_stock_bar1m", "instruments": "cpt_jyc_2026_instruments"}
STATS = ["L_lab10", "M_q_lin", "W_vdir", "W_zero", "Z_mfr60", "Z_mfr90",
         "Y_vd5", "Y_vsd5", "Y_z5", "Z_mfr5d", "Z_net5d", "X_q5d"]


def main():
    txt = SRC.read_text(encoding="utf-8")
    txt = txt.replace("def main(datasources, start_date, end_date):",
                      "def _root(datasources, start_date, end_date, STAT, _cache):", 1)
    txt = re.sub(r'\n    STAT = "[A-Za-z0-9_]+".*\n', "\n", txt, count=1)
    txt = re.sub(r'\n    _cache = \{\}\n', "\n", txt, count=1)
    m = types.ModuleType("mroot")
    exec(compile(txt, str(SRC), "exec"), m.__dict__)
    cache = {}
    cols = {}
    for st in STATS:
        if st.startswith("L_"):
            continue                                   # 季节性根走另一段 SQL，这里只看分钟族
        d = m._root(DS, "2024-01-01 00:00:00", "2024-12-31 23:59:59", st, cache)
        d["date"] = pd.to_datetime(d["date"])
        d["instrument"] = d["instrument"].astype(str)
        d["r"] = d.groupby("date")["factor"].rank(pct=True)
        cols[st] = d.set_index(["date", "instrument"])["r"]
    df = pd.DataFrame(cols)
    print("样本 %d 行（2024 全年，%d 个截面）" % (len(df), df.index.get_level_values(0).nunique()))
    c = df.groupby(level=0).corr(method="spearman").groupby(level=1).mean()
    pd.set_option("display.width", 200)
    print(c.round(3).to_string())


if __name__ == "__main__":
    main()
