# 两个 run 的预测（2023 = val_pred、2024 = pred_2024）在非 09:30 截面上的平均截面秩相关。用法：python e2e/scripts/seed_corr.py runA runB [runC runD ...]（两两成对）
import sys
import pandas as pd
a = sys.argv[1:]
for i in range(0, len(a), 2):
    for fn in ("val_pred.parquet", "pred_2024.parquet"):
        x = pd.read_parquet(f"e2e/runs/{a[i]}/{fn}")[["date", "instrument", "factor"]]
        y = pd.read_parquet(f"e2e/runs/{a[i + 1]}/{fn}")[["date", "instrument", "factor"]]
        m = x.merge(y, on=["date", "instrument"])
        m = m[m["date"].dt.strftime("%H%M") != "0930"]
        c = m.groupby("date")[["factor_x", "factor_y"]].corr(method="spearman").xs("factor_x", level=1)["factor_y"].mean()
        print(f"{a[i]} vs {a[i + 1]} {fn}: {c:.4f}")
