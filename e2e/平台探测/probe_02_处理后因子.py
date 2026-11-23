# -*- coding: utf-8 -*-
# 平台探测 02（2026-10-11）：导出 jyc_eval 处理后的因子值（去极值、标准化、风格剔除之后），
# 用来查清本地与平台剩下的两处小差异（本地 IC 均值高约 6%、多空夏普差约 4%）。
# 需要 probe_01 单元 3 用过的 90 / 104 号 .py 仍在当前目录。只做自测，不提交。
# 产出（用于本地核对，不随仓库发布）：probe2_proc_<编号>.parquet、probe2_info.txt

# %% 单元 1
import contextlib
import glob
import importlib.util
import io
import pandas as pd
from bigmodule import M

START, END = "2024-01-02 00:00:00", "2024-03-01 23:59:59"
DS = {"bar1m": "cpt_jyc_2026_stock_bar1m"}


def load_main(path):
    spec = importlib.util.spec_from_file_location("fac_" + path.split("_")[0], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.main


info = []
for fid in ["90", "104"]:
    path = sorted(glob.glob(f"{fid}_*.py"))[0]
    df = load_main(path)(DS, START, END)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        res = M.jyc_eval._latest(factor_data=df, show=False)
    info.append(f"## {fid} type={type(res)} keys={list(res.keys()) if hasattr(res, 'keys') else None}")
    for k in (res.keys() if hasattr(res, "keys") else []):
        v = res[k]
        info.append(f"  {k}: {type(v)} {getattr(v, 'shape', '')} {list(getattr(v, 'columns', []))[:20]}")
        if isinstance(v, pd.DataFrame):
            v.to_parquet(f"probe2_{k}_{fid}.parquet")
    info.append(buf.getvalue()[-3000:])
open("probe2_info.txt", "w", encoding="utf-8").write("\n".join(info))
print("\n".join(info)[:4000])
