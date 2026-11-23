# 平台口径对比（官方标签 + 官方风格剔除）：候选模型（原始 / 输出中性化 1.0 / 1.5）vs E03 配置 vs 因子 104（本地因子库实现）。
# 用法：YEAR=2023|2024 bash e2e/scripts/remote.sh e2e/scripts/cmp_e04.py 名称=预测.parquet ...
import os, sys, pandas as pd
sys.path.insert(0, "e2e/scripts"); sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission"); sys.path.insert(0, "工具/本地评分")
import analyze_preds as A
import Transformer_modelsave_train as T
YEAR = os.environ.get("YEAR", "2023")
runs = [a.split("=") for a in sys.argv[1:]] or [["E04A", "e2e/runs/E04A_experts_peer/val_pred.parquet"]]
cands = {}
for nm, p in runs + [["E03cfg", "e2e/runs/B1_rev_pool_lneut/" + ("val_pred.parquet" if YEAR == "2023" else f"pred_{YEAR}.parquet")]]:
    d = pd.read_parquet(p)
    if not all(c in d for c in T.CHAR_COLS):  # 风格特征只由行情算出，与 E04A 相同，缺失时从 E04A 补
        ref = pd.read_parquet("e2e/runs/E04A_experts_peer/" + ("val_pred.parquet" if YEAR == "2023" else f"pred_{YEAR}.parquet"))[["date", "instrument"] + T.CHAR_COLS]
        d = d.merge(ref, on=["date", "instrument"], how="left")
    cands[nm + " raw"] = d[["date", "instrument", "factor"]]
    if all(c in d for c in T.CHAR_COLS):
        for fr in (1.0, 1.5):
            cands[f"{nm} n{fr}"] = d[["date", "instrument"]].assign(
                factor=T.neutralize(d["factor"].to_numpy(), d[T.CHAR_COLS].to_numpy(), d["date"].values, fr))
cands["F104"] = pd.read_parquet(f"e2e/runs/analysis/F104_{YEAR}.parquet")[["date", "instrument", "factor"]]
rows = []
for k, f in cands.items():
    t = A.analyze(f, f"{YEAR}-01", f"{YEAR}-12", f"cmp_{k.replace(' ', '_')}_{YEAR}", refs=())
    r = t.loc["E2E"]
    rows.append({"cand": k, **{c: r[c] for c in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}})
pd.set_option("display.width", 250)
print(f"== 平台口径，{YEAR}"); print(pd.DataFrame(rows).set_index("cand").round(4).to_string())
