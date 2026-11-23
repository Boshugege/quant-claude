# 诊断：因子赛道的信号里有没有端到端模型没学到的信息（只推理、不训练，不作为交付）。
# 端到端预测（fold-A 模型，2023 = val_pred、2024 = pred_2024）与评测机上的因子输出逐截面混合：
#   混合分 = (1 − w)·rank(端到端原始分) + w·rank(因子)，之后与 cmp_runs.py 完全相同：输出中性化 1.0 + 09:30 置零 + 平台口径评测。
# 用法：python e2e/scripts/blend_factor.py <run_dir> <名称=factor_keys.parquet> ...   结果写 e2e/runs/analysis/blend_<TAG>.csv
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/scripts"); sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission"); sys.path.insert(0, "工具/本地评分")
import analyze_preds as A
import Transformer_modelsave_train as T

run, facs = sys.argv[1], [a.split("=") for a in sys.argv[2:]]
TAG = os.environ.get("TAG", "factor")
WS = [float(x) for x in os.environ.get("WS", "0,0.1,0.2,0.3,0.5,1").split(",")]
rows = []


def cs_rank(df, col):
    return df.groupby("date")[col].rank(pct=True) - 0.5


for year, fn in (("2023", "val_pred.parquet"), ("2024", "pred_2024.parquet")):
    e = pd.read_parquet(os.path.join(run, fn))
    e["re"] = cs_rank(e, "factor")
    for nm, path in facs:
        f = pd.read_parquet(path)
        f = f.rename(columns={"key_date": "date", "factor": "f"})
        f["instrument"] = f["instrument_id"].map(lambda i: "ID%04d" % int(i))
        f = f[f["date"].dt.year == int(year)][["date", "instrument", "f"]]
        d = e.merge(f, on=["date", "instrument"], how="left")
        cover = float(d["f"].notna().mean())
        d["f"] = d["f"].fillna(0.0)
        d["rf"] = cs_rank(d, "f")
        nz = d["date"].dt.strftime("%H%M") != "0930"
        rho = d[nz].groupby("date")[["re", "rf"]].corr().xs("re", level=1)["rf"].mean()
        for w in WS:
            g = (1 - w) * d["re"].to_numpy() + w * d["rf"].to_numpy()
            g = T.neutralize(g, d[T.CHAR_COLS].to_numpy(), d["date"].values, 1.0)
            g = T.zero_0930(g, d["date"])
            t = A.analyze(d[["date", "instrument"]].assign(factor=g), f"{year}-01", f"{year}-12", f"blend_{TAG}_{nm}_{w:g}_{year}", refs=())
            r = t.loc["E2E"]
            rows.append({"year": year, "factor": nm, "w": w, "corr": round(float(rho), 3), "cover": round(cover, 3),
                         **{c: r[c] for c in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}})
            print(rows[-1], flush=True)
out = pd.DataFrame(rows)
out.to_csv(f"e2e/runs/analysis/blend_{TAG}.csv", index=False)
pd.set_option("display.width", 250)
print(out.round(4).to_string())
