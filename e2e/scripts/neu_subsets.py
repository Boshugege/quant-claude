# 10-25：推理端输出中性化只对部分特征做（c_size、c_ivol 接近官方风格，评分机本来就剔；c_illiq、c_lowp 不是官方风格，评分机会保留并计分），
# 在已有预测上比较（不用重训）。用法：python e2e/scripts/neu_subsets.py <run 目录>... ，结果用 cmp_runs NEU=0 评。
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
SUBS = {"n4": ["c_size", "c_ivol", "c_illiq", "c_lowp"], "n2": ["c_size", "c_ivol"], "n3": ["c_size", "c_ivol", "c_illiq"], "n2l": ["c_size", "c_ivol", "c_lowp"], "n0": []}
for run in sys.argv[1:]:
    for tag, cols in SUBS.items():
        out = run.rstrip("/") + "_" + tag
        os.makedirs(out, exist_ok=True)
        for fn in ("val_pred.parquet", "pred_2024.parquet"):
            d = pd.read_parquet(os.path.join(run, fn))
            if cols:
                d["factor"] = T.neutralize(d["factor"].to_numpy(), d[cols].to_numpy(), d["date"].values, 1.0)
            else:
                d["factor"] = d.groupby("date")["factor"].rank(pct=True) - 0.5
            d.to_parquet(os.path.join(out, fn))
        print(out, flush=True)
