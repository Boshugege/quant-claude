# c129（10-25，诊断）：端到端 R22a 预测里残留多少官方风格／行业暴露，评分机剔除后 IC 损失多少，哪些风格。
import numpy as np, pandas as pd
E = "/srv/media/jyc_scorer/incoming/e2eslot"
STY = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
I2R = dict(zip(_m["local_id"], _m["instrument"]))
p = pd.concat([pd.read_parquet(f"{E}/R22a_val_pred.parquet"), pd.read_parquet(f"{E}/R22a_pred_2024.parquet")])
p["date"] = pd.to_datetime(p["date"]); p = p[p["date"].dt.strftime("%H%M") != "0930"]
p["day"] = p["date"].dt.normalize(); p["real"] = p["instrument"].map(I2R)
ex = pd.concat([pd.read_parquet(f"/home/user/jyc_scorer/data/exposure/exposure_{y}.parquet") for y in (2023, 2024)])
ex["day"] = pd.to_datetime(ex["date"]).dt.normalize()
IND = [c for c in ex.columns if c.isupper() and c not in STY + ["industry_level1_code"]] 
IND = [c for c in IND if ex[c].dropna().isin([0, 1]).all()]
x = p.merge(ex[["day", "instrument"] + STY + IND].rename(columns={"instrument": "real"}), on=["day", "real"], how="inner")
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"])
x = x.merge(lab, on=["date", "instrument"], how="left").dropna(subset=["y"])
print("样本", len(x), "行业哑变量", len(IND))
res = []
for d, g in x.groupby("date"):
    f = g["factor"].rank(pct=True).to_numpy() - 0.5
    yr = g["y"].rank(pct=True).to_numpy() - 0.5
    A = np.column_stack([np.ones(len(g)), g[STY].to_numpy(float), g[IND].to_numpy(float)])
    ok = np.isfinite(A).all(1)
    if ok.sum() < 100: continue
    b, *_ = np.linalg.lstsq(A[ok], f[ok], rcond=None)
    fit = A[ok] @ b; rs = f[ok] - fit
    As = np.column_stack([np.ones(ok.sum()), g[STY].to_numpy(float)[ok]])
    bs, *_ = np.linalg.lstsq(As, f[ok], rcond=None)
    r2s = 1 - ((f[ok] - As @ bs) ** 2).sum() / ((f[ok] - f[ok].mean()) ** 2).sum()
    ic = lambda v: np.corrcoef(v, yr[ok])[0, 1]
    res.append(dict(date=d, r2=1 - (rs ** 2).sum() / ((f[ok] - f[ok].mean()) ** 2).sum(), r2_sty=r2s,
                    ic_raw=ic(f[ok]), ic_res=ic(rs), ic_fit=ic(fit), **{k: bs[i + 1] for i, k in enumerate(STY)}))
R = pd.DataFrame(res); R["yr"] = pd.DatetimeIndex(R["date"]).year
for yy, g in R.groupby("yr"):
    print(yy, "风格+行业解释预测方差 R² %.3f（只风格 %.3f）；IC 原始 %.4f → 剔除后 %.4f；被剔除部分自身 IC %.4f" % (g["r2"].mean(), g["r2_sty"].mean(), g["ic_raw"].mean(), g["ic_res"].mean(), g["ic_fit"].mean()))
print("各风格系数（预测的截面秩对风格 z 的回归，均值／t）：")
for k in STY:
    v = R[k]; print("  %-9s %+.4f  t %+.1f" % (k, v.mean(), v.mean() / v.std() * np.sqrt(len(v))))
