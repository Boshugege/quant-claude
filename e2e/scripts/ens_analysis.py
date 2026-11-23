# 集成收益来源分析（只推理、不训练）：fold-A 各模型预测之间的截面秩相关矩阵，以及若干集成组合的平台口径五项。
# 用法：python e2e/scripts/ens_analysis.py   （模型清单与组合写在下面）；结果写 e2e/runs/analysis/ens_analysis_*.csv
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/scripts"); sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission"); sys.path.insert(0, "工具/本地评分")
import analyze_preds as A
import Transformer_modelsave_train as T

R = "e2e/runs/"
RUNS = {"F42": "S12_A2_fuse", "F43": "S17z_A2_seed43",                     # 融合头（E11 配方）两个种子
        "N42": "S8c_aux05", "N43": "S10z_seed43", "N44": "S11z_seed44",      # 无融合头（E09 配方）三个种子
        "Fh": "S17a_fuse_h256", "Fp": "S17b_fuse_peer", "Fw": "S17c_fuse_headw05"}   # 融合头的三个结构变体（种子 42）
COMBOS = {"F42": ["F42"], "F42+F43": ["F42", "F43"], "N42+N43+N44": ["N42", "N43", "N44"],
          "F42+N43": ["F42", "N43"], "F42+F43+N42+N43": ["F42", "F43", "N42", "N43"],
          "F42+Fh+Fp+Fw": ["F42", "Fh", "Fp", "Fw"], "F42+F43+Fh+Fp": ["F42", "F43", "Fh", "Fp"],
          "全部8个": list(RUNS)}
rows, cors = [], []
for year, fn in (("2023", "val_pred.parquet"), ("2024", "pred_2024.parquet")):
    base = None
    for k, r in RUNS.items():
        d = pd.read_parquet(os.path.join(R, r, fn))
        d = d[d["date"].dt.strftime("%H%M") != "0930"] if False else d
        d[k] = d.groupby("date")["factor"].rank(pct=True) - 0.5
        keep = ["date", "instrument", k] + ([c for c in T.CHAR_COLS] if base is None else [])
        base = d[keep] if base is None else base.merge(d[["date", "instrument", k]], on=["date", "instrument"], how="inner")
    nz = base["date"].dt.strftime("%H%M") != "0930"
    ks = list(RUNS)
    C = base[nz].groupby("date")[ks].corr().groupby(level=1).mean().loc[ks, ks]
    cors.append(C.round(3).assign(year=year))
    for name, members in COMBOS.items():
        f = base[members].mean(1).to_numpy()
        f = T.neutralize(f, base[T.CHAR_COLS].to_numpy(), base["date"].values, 1.0)
        f = T.zero_0930(f, base["date"])
        t = A.analyze(base[["date", "instrument"]].assign(factor=f), f"{year}-01", f"{year}-12", f"ens_{name}_{year}", refs=())
        rr = t.loc["E2E"]
        rows.append({"year": year, "combo": name, "n": len(members), **{c: rr[c] for c in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}})
        print(rows[-1], flush=True)
pd.set_option("display.width", 250)
cm = pd.concat(cors)
print("== 预测之间的截面秩相关（非 09:30 截面均值）"); print(cm.to_string())
out = pd.DataFrame(rows)
print("== 集成组合（平台口径）"); print(out.round(4).to_string())
os.makedirs("e2e/runs/analysis", exist_ok=True)
cm.to_csv("e2e/runs/analysis/ens_analysis_corr.csv"); out.to_csv("e2e/runs/analysis/ens_analysis_combo.csv", index=False)
