# 不训练：把已有单模型预测做截面排名平均，模拟"多种子集成"对五项指标（尤其换手）的影响。平台口径 1.0 倍中性化 + 09:30 置零。
import sys
import pandas as pd
sys.path.insert(0, "e2e/scripts"); sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission"); sys.path.insert(0, "工具/本地评分")
import analyze_preds as A
import Transformer_modelsave_train as T
SETS = {"G3": ["G3_e07_proxyind"], "H1": ["H1_skip930"], "H4": ["H4_winfeat7"],
        "G3+H1": ["G3_e07_proxyind", "H1_skip930"],
        "H1+H3+H4+H5": ["H1_skip930", "H3_icir", "H4_winfeat7", "H5_peer240"]}
rows = []
for year, fn in (("2023", "val_pred.parquet"), ("2024", "pred_2024.parquet")):
    cache = {}
    for nm, runs in SETS.items():
        ds = []
        for r in runs:
            if r not in cache:
                cache[r] = pd.read_parquet(f"e2e/runs/{r}/{fn}")
            d = cache[r]
            ds.append(d.set_index(["date", "instrument"])["factor"].groupby(level=0).rank(pct=True))
        base = cache[runs[0]].set_index(["date", "instrument"])
        f = pd.concat(ds, axis=1).mean(axis=1).reindex(base.index)
        for neu in (1.0, 0.0):
            g = T.neutralize(f.to_numpy(), base[T.CHAR_COLS].to_numpy(), base.index.get_level_values(0).values, neu) if neu > 0 else f.to_numpy()
            g = T.zero_0930(g, pd.Series(base.index.get_level_values(0)))
            t = A.analyze(base.reset_index()[["date", "instrument"]].assign(factor=g), f"{year}-01", f"{year}-12", f"ens_{nm}_{neu}_{year}", refs=())
            r = t.loc["E2E"]
            rows.append({"year": year, "set": nm, "n": len(runs), "neu": neu, **{c: round(float(r[c]), 4) for c in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}})
            print(rows[-1], flush=True)
out = pd.DataFrame(rows); out.to_csv("e2e/runs/analysis/ens_check.csv", index=False)
print(out.to_string())
