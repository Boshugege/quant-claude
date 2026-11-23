# c112（10-24）：把端到端 R22a 的 2023／2024 预测按平台口径（输出中性化 1 + 09:30 置零）交评分机，取逐截面 IC，与因子组合 1663 对比。
import sys, subprocess, numpy as np, pandas as pd
D = "/srv/media/jyc_scorer/incoming/e2eslot"
CH = ["c_size", "c_ivol", "c_illiq", "c_lowp"]

def neutralize(values, chars, dates, frac=1.0):
    d = pd.Series(dates)
    y = pd.Series(np.asarray(values, np.float64)).groupby(d.values).rank(pct=True) - 0.5
    X = np.column_stack([(pd.Series(chars[:, j]).groupby(d.values).rank(pct=True) - 0.5).fillna(0.0).to_numpy() for j in range(chars.shape[1])])
    ok = y.notna().to_numpy()
    codes, uniq = pd.factorize(d.values)
    k, G = X.shape[1], len(uniq)
    Xc = np.column_stack([np.ones(len(X)), X]) * ok[:, None]
    yv = np.nan_to_num(y.to_numpy()) * ok
    XtX = np.zeros((G, k + 1, k + 1)); Xty = np.zeros((G, k + 1))
    for i in range(k + 1):
        Xty[:, i] = np.bincount(codes, Xc[:, i] * yv, minlength=G)
        for j in range(i, k + 1):
            w = np.bincount(codes, Xc[:, i] * Xc[:, j], minlength=G); XtX[:, i, j] = w; XtX[:, j, i] = w
    b = np.linalg.solve(XtX + 1e-9 * np.eye(k + 1), Xty[..., None])[..., 0]
    return np.where(ok, y.to_numpy() - frac * (X * b[codes, 1:]).sum(axis=1), np.nan)

for src, yr in [("R22a_val_pred", "2023"), ("R22a_pred_2024", "2024")]:
    p = pd.read_parquet(f"{D}/{src}.parquet")
    p["date"] = pd.to_datetime(p["date"])
    f = neutralize(p["factor"].to_numpy(), p[CH].to_numpy(np.float64), p["date"].to_numpy())
    f[p["date"].dt.strftime("%H%M").to_numpy() == "0930"] = 0.0
    out = f"{D}/e2e_R22a_{yr}.parquet"
    pd.DataFrame({"date": p["date"], "instrument": p["instrument"], "factor": f}).to_parquet(out, index=False)
    r = subprocess.run(["curl", "-s", "-F", f"files=@{out}", "-F", "kind=values", "-F", f"start={yr}-01", "-F", f"end={yr}-12",
                        "-F", f"name=e2eslot_R22a_{yr}", "http://127.0.0.1:46220/api/jobs"], capture_output=True, text=True)
    print(yr, r.stdout[:200])
