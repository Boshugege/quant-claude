# 平台口径对比多个训练（两折：2023 = val_pred，2024 = pred_2024），推理口径 = 输出中性化 NEU（默认 1.0）+ 09:30 置零（ZERO=1）。
# 用法：bash e2e/scripts/remote.sh e2e/scripts/cmp_runs.py 名称=run_dir ...   结果写 e2e/runs/analysis/cmp_<TAG>.csv
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/scripts"); sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission"); sys.path.insert(0, "工具/本地评分")
import analyze_preds as A
import Transformer_modelsave_train as T
NEU, ZERO, TAG = float(os.environ.get("NEU", "1.0")), os.environ.get("ZERO", "1") == "1", os.environ.get("TAG", "runs")
OUTZ, OUTZ_CLIP = os.environ.get("OUTZ", "0") == "1", float(os.environ.get("OUTZ_CLIP", "5"))
rows = []
PAIRS = [tuple(a.split(":")) for a in os.environ.get("PAIRS", "2023:val_pred.parquet,2024:pred_2024.parquet").split(",")]
for year, fn in PAIRS:
    for nm, run in (a.split("=") for a in sys.argv[1:]):
        p = os.path.join(run, fn)
        if not os.path.exists(p):
            print("缺少", p); continue
        d = pd.read_parquet(p)
        v = d["factor"].to_numpy()
        if OUTZ:                                                                  # 10-23：输出改用逐截面 z 分数（不转均匀排名）再中性化
            g = pd.Series(v).groupby(d["date"].values)
            v = ((pd.Series(v) - g.transform("mean")) / g.transform("std").replace(0, np.nan)).clip(-OUTZ_CLIP, OUTZ_CLIP).fillna(0.0).to_numpy()
        f = T.neutralize(v, d[T.CHAR_COLS].to_numpy(), d["date"].values, NEU, rank_values=not OUTZ) if NEU > 0 else v
        if ZERO:
            f = T.zero_0930(f, d["date"])
        t = A.analyze(d[["date", "instrument"]].assign(factor=f), f"{year}-01", f"{year}-12", f"cmp_{TAG}_{nm}_{year}", refs=())
        r = t.loc["E2E"]
        rows.append({"year": year, "run": nm, **{c: r[c] for c in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}})
out = pd.DataFrame(rows)
out.to_csv(f"e2e/runs/analysis/cmp_{TAG}.csv", index=False)
pd.set_option("display.width", 250)
print(f"== 平台口径（输出中性化 {NEU:g}{' + 09:30 置零' if ZERO else ''}）")
print(out.round(4).to_string())
