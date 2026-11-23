# E07 S0：行情代理特征与伪行业能替代多少官方风格 / 行业（只用于评估，提交代码不读官方表）。
# 每个交易日做截面回归：官方风格 ~ [1, 代理特征(+伪行业)]，报平均 R²；伪行业与申万一级行业的调整兰德指数（ARI）。
# 用法：bash e2e/scripts/remote.sh e2e/scripts/s0_proxy_check.py 2023-01-01 2023-12-31
import sys, time
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend, exposure_local, IND_COLS

sd, ed = (sys.argv[1], sys.argv[2]) if len(sys.argv) > 2 else ("2023-01-01", "2023-12-31")
fb = FeatherBackend()
inst = fb.pool(sd, ed + " 23:59:59")
t0 = time.time()
pt = T.proxy_table(fb, sd, ed + " 23:59:59", inst)
print(f"proxy_table {len(pt):,} 行，{time.time() - t0:.0f}s；覆盖率", pt[T.P_COLS[:1] + ["pind"]].notna().mean().round(3).to_dict(),
      "伪行业已分配", round(float((pt["pind"] >= 0).mean()), 3))
STY = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]
ex = exposure_local(sd, ed + " 23:59:59", inst).rename(columns={"date": "day"})
ex["day"] = pd.to_datetime(ex["day"]).dt.normalize()
d = pt.merge(ex, on=["day", "instrument"], how="inner").dropna(subset=STY)
cols = [c for c in T.P_COLS if c != "p_size2"]


def r2_by_day(g, use_ind):
    Z = g[cols].to_numpy(np.float64)
    Z = (Z - np.nanmean(Z, 0)) / (np.nanstd(Z, 0) + 1e-12)
    Z = np.nan_to_num(np.clip(Z, -5, 5))
    X = [np.ones((len(g), 1)), Z, (Z[:, [cols.index("p_size")]] ** 2)]
    if use_ind:
        oh = np.zeros((len(g), T.PIND_K))
        ok = g["pind"].to_numpy() >= 0
        oh[np.flatnonzero(ok), g["pind"].to_numpy()[ok].astype(int)] = 1
        X.append(oh)
    X = np.concatenate(X, 1)
    out = {}
    for s in STY + ["IND"]:
        if s == "IND":     # 行业：申万一级哑变量能被解释的平均比例（逐个行业 R² 按市值无关等权平均）
            Y = g[IND_COLS].to_numpy(np.float64)
        else:
            Y = g[[s]].to_numpy(np.float64)
        coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
        res = Y - X @ coef
        tot = ((Y - Y.mean(0)) ** 2).sum(0)
        r2 = 1 - (res ** 2).sum(0) / np.where(tot > 0, tot, np.nan)
        out[s] = float(np.nanmean(r2))
    return out


days = sorted(d["day"].unique())[::5]       # 每 5 个交易日取一天，足够估计平均 R²
rows = {False: [], True: []}
for dy in days:
    g = d[d["day"] == dy]
    if len(g) < 300:
        continue
    for u in (False, True):
        rows[u].append(r2_by_day(g, u))
pd.set_option("display.width", 250)
print("官方风格 / 行业被行情代理解释的平均截面 R²（每 5 个交易日抽一天）")
print(pd.DataFrame({"代理": pd.DataFrame(rows[False]).mean(), "代理+伪行业": pd.DataFrame(rows[True]).mean()}).round(3).T.to_string())

def adjusted_rand_score(a, b):
    """调整兰德指数（不依赖 sklearn）。"""
    a, b = pd.factorize(np.asarray(a))[0], pd.factorize(np.asarray(b))[0]
    c = np.zeros((a.max() + 1, b.max() + 1))
    np.add.at(c, (a, b), 1)
    comb = lambda x: x * (x - 1) / 2  # noqa: E731
    sij, sa, sb, n = comb(c).sum(), comb(c.sum(1)).sum(), comb(c.sum(0)).sum(), comb(len(a))
    exp = sa * sb / n
    return (sij - exp) / (0.5 * (sa + sb) - exp)


sw = d[IND_COLS].to_numpy().argmax(1)
ari = [adjusted_rand_score(sw[(d["day"] == dy).to_numpy() & (d["pind"] >= 0).to_numpy()],
                           d.loc[(d["day"] == dy) & (d["pind"] >= 0), "pind"]) for dy in days]
print(f"伪行业与申万一级行业的调整兰德指数 ARI：均值 {np.nanmean(ari):.3f}（0 = 无关，1 = 完全一致）")
