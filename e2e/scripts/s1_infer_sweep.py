# E07 S1：只改推理的两项杠杆（不重训），用 E06A 两折的现成预测（2020–22 训练；2023 = val_pred，2024 = pred_2024）。
#   - 输出中性化力度 0 / 1.0 / 1.5 / 2.0 / 2.5（四特征，同 E05 推理）；
#   - 09:30 截面输出常数（平台跳过常数截面；因子赛道 104 号置零后 IC_IR、换手都更好）。
# 平台口径（官方标签 + 10 风格 + 32 行业剔除）。用法：bash e2e/scripts/remote.sh e2e/scripts/s1_infer_sweep.py [run_dir]
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/scripts"); sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission"); sys.path.insert(0, "工具/本地评分")
import analyze_preds as A
import Transformer_modelsave_train as T

RUN = sys.argv[1] if len(sys.argv) > 1 else "e2e/runs/E06A_experts_peer_barra_yoff"
FRACS = (0.0, 1.0, 1.5, 2.0, 2.5)
rows = []
for year, fn in (("2023", "val_pred.parquet"), ("2024", "pred_2024.parquet")):
    d = pd.read_parquet(os.path.join(RUN, fn))
    is930 = (d["date"].dt.strftime("%H%M") == "0930").to_numpy()
    for fr in FRACS:
        f = d["factor"].to_numpy() if fr == 0 else T.neutralize(d["factor"].to_numpy(), d[T.CHAR_COLS].to_numpy(), d["date"].values, fr)
        for z in (False, True):
            g = np.where(is930, 0.0, f) if z else f
            tag = f"s1_n{fr}_{'z930' if z else 'k930'}_{year}"
            t = A.analyze(d[["date", "instrument"]].assign(factor=g), f"{year}-01", f"{year}-12", tag, refs=())
            r = t.loc["E2E"]
            rows.append({"year": year, "neutral": fr, "zero_0930": z, "n_xsec": None,
                         **{c: r[c] for c in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")},
                         **{c: r[c] for c in t.columns if c.startswith("s")}})
out = pd.DataFrame(rows)
os.makedirs("e2e/runs/analysis", exist_ok=True)
out.to_csv("e2e/runs/analysis/s1_infer_sweep.csv", index=False)
pd.set_option("display.width", 250)
print(out.drop(columns=["n_xsec"]).round(4).to_string())
