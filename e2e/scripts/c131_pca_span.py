# 10-25（诊断）：用过去 240 个交易日超额日收益的主成分载荷（每月初重算，只用之前的数据）作为统计风险因子，加进行情代理，
# 看对官方风格、申万行业的覆盖能提高多少。官方暴露只用来量覆盖度，不进训练／推理。
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
dp = T.daily_panel(fb, "2021-12-01 00:00:00", ed, inst)
piv = lambda c: dp.pivot(index="day", columns="instrument", values=c).sort_index()
close, pre = piv("close"), piv("pre")
r = np.log(close / pre).where(lambda v: np.isfinite(v) & (v.abs() < 0.5))
ex_r = r.sub(r.mean(axis=1), axis=0)
days = ex_r.index
months = pd.Series(days[days >= pd.Timestamp(sd)]).groupby(pd.Series(days[days >= pd.Timestamp(sd)]).dt.to_period("M")).first()
KS = (10, 20, 30)
load = {k: [] for k in KS}
for d0 in months:
    i0 = days.get_loc(d0)
    blk = ex_r.iloc[max(0, i0 - 240):i0]
    ok = blk.notna().sum() >= 150
    Z = blk.loc[:, ok]
    Z = (Z - Z.mean()) / (Z.std() + 1e-12)
    Z = Z.fillna(0.0).to_numpy()
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    mdays = days[(days >= d0) & (days < d0 + pd.offsets.MonthBegin(1))]
    for k in KS:
        L = pd.DataFrame(Vt[:k].T * S[:k], index=blk.columns[ok], columns=[f"pc{j}" for j in range(k)])
        for d in mdays:
            load[k].append(L.assign(day=d).reset_index().rename(columns={"index": "instrument"}))
STY = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
ex = pd.concat([pd.read_parquet(f"数据/exposure/exposure_{y}.parquet") for y in (2023, 2024)])
ex["day"] = pd.to_datetime(ex["date"]).dt.normalize()
mp = pd.read_csv("数据/exposure/id_map.csv", dtype=str); mp = mp[mp["ok"].str.lower() == "true"]
ex = ex.merge(mp[["instrument", "local_id"]], on="instrument").drop(columns=["instrument"]).rename(columns={"local_id": "instrument"})
base = pt.merge(ex[["day", "instrument", "industry_level1_code"] + STY], on=["day", "instrument"], how="inner")
def span(x, cols, use_pind):
    out = {k: [] for k in STY}; ind = []
    for d in sorted(x["day"].unique())[::5]:
        g = x[x["day"] == d]
        Z = g[cols].to_numpy(float); Z = (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-12); Z = np.nan_to_num(Z)
        A = np.column_stack([np.ones(len(g)), Z, Z[:, [cols.index("p_size")]] ** 2])
        if use_pind:
            A = np.column_stack([A, pd.get_dummies(g["pind"].astype(int)).to_numpy(float)])
        for k in STY:
            yv = g[k].to_numpy(float); ok = np.isfinite(yv)
            b, *_ = np.linalg.lstsq(A[ok], yv[ok], rcond=None); out[k].append(1 - (yv[ok] - A[ok] @ b).var() / yv[ok].var())
        I = pd.get_dummies(g["industry_level1_code"]).to_numpy(float)
        b, *_ = np.linalg.lstsq(A, I, rcond=None); rr = I - A @ b
        ind.append(1 - (rr ** 2).sum() / ((I - I.mean(0)) ** 2).sum())
    return {k: np.mean(v) for k, v in out.items()}, np.mean(ind)
res = {"现有（代理 + 伪行业）": span(base, P, True)}
for k in KS:
    x = base.merge(pd.concat(load[k]), on=["day", "instrument"], how="left")
    pc = [f"pc{j}" for j in range(k)]
    res[f"代理 + 主成分 {k}"] = span(x, P + pc, False)
    res[f"代理 + 伪行业 + 主成分 {k}"] = span(x, P + pc, True)
print("%-24s " % "" + " ".join("%8s" % s[:8] for s in STY) + "    行业")
for nm, (st, ind) in res.items():
    print("%-24s " % nm + " ".join("%8.3f" % st[s] for s in STY) + "   %.3f" % ind)
