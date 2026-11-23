# 10-25（诊断，只用于设计训练用的行情代理；官方暴露表不进训练／推理）：端到端训练用的行情代理（proxy_table）对官方 10 风格、申万行业的覆盖程度。
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend
fb = FeatherBackend()
sd, ed = "2023-01-01 00:00:00", "2024-12-31 23:59:59"
inst = fb.pool(sd, ed)
pt = T.proxy_table(fb, sd, ed, inst)
P = [c for c in pt.columns if c.startswith("p_") and c != "p_size2"]
print("代理列", P, "伪行业列", [c for c in pt.columns if not c.startswith("p_") and c not in ("day", "instrument")][:3])
STY = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
ex = pd.concat([pd.read_parquet(f"数据/exposure/exposure_{y}.parquet") for y in (2023, 2024)])
ex["day"] = pd.to_datetime(ex["date"]).dt.normalize()
mp = pd.read_csv("数据/exposure/id_map.csv", dtype=str); mp = mp[mp["ok"].str.lower() == "true"]
ex = ex.merge(mp[["instrument", "local_id"]], on="instrument").drop(columns=["instrument"]).rename(columns={"local_id": "instrument"})
x = pt.merge(ex[["day", "instrument", "industry_level1_code"] + STY], on=["day", "instrument"], how="inner")
pind = [c for c in pt.columns if c in ("pind", "p_ind", "pseudo_ind")]
print("合并行", len(x), "伪行业列", pind)
days = sorted(x["day"].unique())[::5]
out = {k: [] for k in STY}; ind_r2 = []
for d in days:
    g = x[x["day"] == d]
    Z = g[P].to_numpy(float); Z = (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-12); Z = np.nan_to_num(Z)
    A = np.column_stack([np.ones(len(g)), Z, Z[:, [P.index("p_size")]] ** 2])
    if pind:
        D = pd.get_dummies(g[pind[0]].astype(int)).to_numpy(float); A = np.column_stack([A, D])
    for k in STY:
        yv = g[k].to_numpy(float); ok = np.isfinite(yv)
        b, *_ = np.linalg.lstsq(A[ok], yv[ok], rcond=None); r = yv[ok] - A[ok] @ b
        out[k].append(1 - r.var() / yv[ok].var())
    if pind:
        I = pd.get_dummies(g["industry_level1_code"]).to_numpy(float)
        D = pd.get_dummies(g[pind[0]].astype(int)).to_numpy(float)
        b, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(g)), D]), I, rcond=None)
        r = I - np.column_stack([np.ones(len(g)), D]) @ b
        ind_r2.append(1 - (r ** 2).sum() / ((I - I.mean(0)) ** 2).sum())
print("官方风格被行情代理（+伪行业）解释的截面 R²：")
for k in STY: print("  %-9s %.3f" % (k, np.mean(out[k])))
if ind_r2: print("申万一级行业哑变量被伪行业解释的 R² %.3f" % np.mean(ind_r2))
