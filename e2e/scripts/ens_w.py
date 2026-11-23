# 加权集成（各成员截面居中排名后按权重平均），写成 cmp_runs.py 能读的 run 目录。用法：python e2e/scripts/ens_w.py <输出目录> <run1>[:w] <run2>[:w] ...
import os, sys
import pandas as pd
out, specs = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
runs = [(s.rsplit(":", 1)[0], float(s.rsplit(":", 1)[1])) if ":" in os.path.basename(s) else (s, 1.0) for s in specs]
for fn in ("val_pred.parquet", "pred_2024.parquet"):
    m = None
    for i, (r, w) in enumerate(runs):
        d = pd.read_parquet(os.path.join(r, fn))
        d[f"f{i}"] = (d.groupby("date")["factor"].rank(pct=True) - 0.5) * w
        keep = ["date", "instrument", f"f{i}"] + ([c for c in d.columns if c.startswith("c_")] if m is None else [])
        m = d[keep] if m is None else m.merge(d[["date", "instrument", f"f{i}"]], on=["date", "instrument"], how="inner")
    m["factor"] = m[[f"f{i}" for i in range(len(runs))]].sum(1) / sum(w for _, w in runs)
    m.drop(columns=[f"f{i}" for i in range(len(runs))]).to_parquet(os.path.join(out, fn))
    print(fn, len(m))
