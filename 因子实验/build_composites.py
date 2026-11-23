# -*- coding: utf-8 -*-
"""从「头部底座 + 新根因子」生成第 4 级的正式合成版因子文件。

用法：
  /home/user/jyc_scorer/.venv/bin/python 因子实验/build_composites.py \
      <头部文件> <新根文件> <权重 w> <输出文件> <编号> <方向标题>

合成口径与任务书第 3 级（root_eval.py --grid）完全一致：
  B_w = rank(头部最终因子) + w · rank(新根最终因子)（逐截面 pct 秩 − 0.5），
  只保留平台 8 个约定截面，以股票池 (date, instrument) 为骨架、缺失填 0，0930 置 0。
头部代码与新根代码**逐字节原样内联**（只把 def main 改名），因此二者自己算出的因子与
单独交付时完全相同；合成只发生在最后一步。
"""
import os
import re
import sys

HEAD = sys.argv[1]
ROOT = sys.argv[2]
W = float(sys.argv[3])
OUT = sys.argv[4]
NUM = sys.argv[5]
TITLE = sys.argv[6]
MODE = sys.argv[7] if len(sys.argv) > 7 else "raw"      # raw（推荐，不破坏头部分布）/ rank（任务书 §4.2 字面口径）

SLOTS = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}


def extract(path, newname):
    src = open(path, encoding="utf-8").read()
    cut = src.find('if __name__ == "__main__":')
    if cut > 0:
        src = src[:cut]
    i = src.find("def main(")
    if i < 0:
        raise SystemExit("找不到 def main：%s" % path)
    body = src[i:]
    body = body.replace("def main(", "def %s(" % newname, 1)
    return body.rstrip() + "\n"


head_mod = os.path.splitext(os.path.basename(HEAD))[0]
root_mod = os.path.splitext(os.path.basename(ROOT))[0]

header = '''def main(datasources, start_date, end_date):
    """【新根因子第 4 级·正式合成版 %s】%s

    底座：`%s`（头部代码逐字节内联为 `_head_impl`，单独跑出来的因子与底座完全一致）。
    新根：`%s`（同样逐字节内联为 `_root_impl`）。
    合成：B_w = 头部最终因子 + %.2f · rank(新根最终因子)（_HEAD_MODE=raw 时头部保留原始值；rank 时头部也取 pct 秩），
    与任务书第 3 级离线网格（root_eval.py --grid）度量的对象一致；只保留平台 8 个约定截面，
    以股票池为骨架、缺失填 0，0930 置 0。
    """
    import numpy as np
    import pandas as pd
    import dai

    _HEAD_MODE = "%s"
    _head = _head_impl(datasources, start_date, end_date)
    _root = _root_impl(datasources, start_date, end_date)
    _head = _head.rename(columns={"factor": "fh"})
    _root = _root.rename(columns={"factor": "fr"})
    _head["date"] = pd.to_datetime(_head["date"])
    _root["date"] = pd.to_datetime(_root["date"])
    _head["instrument"] = _head["instrument"].astype(str)
    _root["instrument"] = _root["instrument"].astype(str)
    _m = _head.merge(_root, on=["date", "instrument"], how="inner")
    _m = _m[np.isfinite(_m["fh"]) & np.isfinite(_m["fr"])].copy()
    _m["rh"] = _m["fh"] if _HEAD_MODE == "raw" else (_m.groupby("date")["fh"].rank(pct=True) - 0.5)
    _m["rr"] = _m.groupby("date")["fr"].rank(pct=True) - 0.5
    _m["factor"] = _m["rh"].to_numpy() + %.2f * _m["rr"].to_numpy()

    _slots = {"0930", "1000", "1030", "1100", "1130", "1330", "1400", "1430"}
    _inst_tbl = datasources.get("instruments", "cpt_jyc_2026_instruments")
    _pool = dai.query(f"SELECT date, instrument FROM {_inst_tbl}",
                      filters={"date": [start_date, end_date]}, compression=True).df()
    _pool["date"] = pd.to_datetime(_pool["date"])
    _pool["instrument"] = _pool["instrument"].astype(str)
    _pool = _pool[_pool["date"].dt.strftime("%%H%%M").isin(_slots)]
    _pool = _pool.drop_duplicates(["date", "instrument"])
    _out = _pool.merge(_m[["date", "instrument", "factor"]], on=["date", "instrument"], how="left")
    _out["factor"] = _out["factor"].astype("float64").fillna(0.0)
    _out.loc[_out["date"].dt.strftime("%%H%%M") == "0930", "factor"] = 0.0
    return _out[["date", "instrument", "factor"]].sort_values(["date", "instrument"]).reset_index(drop=True)


''' % (NUM, TITLE, head_mod, root_mod, W, MODE, W)

text = header + extract(HEAD, "_head_impl") + "\n\n" + extract(ROOT, "_root_impl")
open(OUT, "w", encoding="utf-8").write(text)
print("写出 %s（%d 行；w=%.2f）" % (OUT, text.count("\n"), W))
