# 压力诊断（10-20）：平台口径下，按截面标签离散度中位数把截面分成高、低两半，各自的 IC 均值、IC 标准差、IC_IR（压力 = 较弱一半的 IC_IR）；
# 再按离散度五档看 IC 均值与截面数。用法：python e2e/scripts/stress_halves.py 名称=run_dir ...（2023 = val_pred，2024 = pred_2024）
import os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/scripts"); sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission"); sys.path.insert(0, "工具/本地评分")
import analyze_preds as A
import localeval as le
import Transformer_modelsave_train as T
for year, fn in (("2023", "val_pred.parquet"), ("2024", "pred_2024.parquet")):
    panel = A.panel_for(f"{year}-01", f"{year}-12")
    for nm, run in (a.split("=") for a in sys.argv[1:]):
        d = pd.read_parquet(os.path.join(run, fn))
        f = T.zero_0930(T.neutralize(d["factor"].to_numpy(), d[T.CHAR_COLS].to_numpy(), d["date"].values, 1.0), d["date"])
        m = panel[["key_date", "instrument"]].merge(d[["date", "instrument"]].assign(factor=f).rename(columns={"date": "key_date"}), how="left", on=["key_date", "instrument"])
        fr = le.style_residual(pd.Series(m["factor"].to_numpy(), index=panel.index), panel)
        ev = panel["is_eval"].values
        x = pd.DataFrame({"date": panel["key_date"].values[ev], "f": fr.values[ev], "y": panel["y_off"].values[ev]}).dropna()
        x["rf"] = x.groupby("date")["f"].rank(pct=True); x["ry"] = x.groupby("date")["y"].rank(pct=True)
        g = x.groupby("date")
        ic = g.apply(lambda q: np.corrcoef(q["rf"], q["ry"])[0, 1] if len(q) >= 100 else np.nan).dropna()
        ysd = g["y"].std().reindex(ic.index)
        hi = ysd > ysd.median()
        ir = lambda s: s.mean() / s.std()
        print(f"{year} {nm}: 全部 IC {ic.mean():.4f} IR {ir(ic):.3f} | 高离散半 IC {ic[hi].mean():.4f} σ {ic[hi].std():.4f} IR {ir(ic[hi]):.3f} | "
              f"低离散半 IC {ic[~hi].mean():.4f} σ {ic[~hi].std():.4f} IR {ir(ic[~hi]):.3f}", flush=True)
        q = pd.qcut(ysd, 5, labels=False)
        print("   离散度五档 IC 均值：" + " ".join(f"{ic[q == k].mean():+.4f}" for k in range(5)) + "  σ：" + " ".join(f"{ic[q == k].std():.4f}" for k in range(5)), flush=True)
