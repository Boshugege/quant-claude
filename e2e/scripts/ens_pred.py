# 两个（或多个）fold-A 模型的预测等权集成（各自截面居中排名后平均），写成 cmp_runs.py 能读的 run 目录。
# 用法：python e2e/scripts/ens_pred.py <输出目录> <run1> <run2> [...]
import os, sys
import pandas as pd
out, runs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
for fn in ("val_pred.parquet", "pred_2024.parquet"):
    ds = [pd.read_parquet(os.path.join(r, fn)) for r in runs]
    m = ds[0].copy()
    m["factor"] = m.groupby("date")["factor"].rank(pct=True) - 0.5
    for i, d in enumerate(ds[1:], 1):
        d = d[["date", "instrument", "factor"]].copy()
        d[f"f{i}"] = d.groupby("date")["factor"].rank(pct=True) - 0.5
        m = m.merge(d[["date", "instrument", f"f{i}"]], on=["date", "instrument"], how="inner")
    m["factor"] = m[["factor"] + [f"f{i}" for i in range(1, len(ds))]].mean(1)
    m.drop(columns=[f"f{i}" for i in range(1, len(ds))]).to_parquet(os.path.join(out, fn))
    print(fn, len(m), [len(d) for d in ds])
