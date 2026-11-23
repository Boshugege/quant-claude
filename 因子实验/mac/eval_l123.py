# -*- coding: utf-8 -*-
"""第 1–3 级快速评估（macstudio 本地，平台口径近似）：
  第 1 级：单独五项（风格 + 行业剔除后）；第 2 级：剔除 490 后的残差 IC（按单独 IC 定向）；
  第 3 级：B = H + s·w·X（X 为截面秩 −0.5～0.5，s 为方向），w 网格，报相对 490 的五项差，标"每项都不变差"。
风格剔除是线性投影，逐截面预先算好正交基，任意组合只需投影一次。用法：python eval_l123.py 1230 1231 ..."""
import os, sys
import numpy as np
import pandas as pd
R = os.path.expanduser("~/Desktop/work/quant")
os.chdir(R)
for p in ("e2e/scripts", "e2e/src", "e2e/submission", "工具/本地评分"):
    sys.path.insert(0, p)
import localeval as le  # noqa: E402
import analyze_preds as A  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
WS = [float(x) for x in os.environ.get("WS", "0.05,0.10,0.20,0.30").split(",")]
names = sys.argv[1:]
k = pd.read_parquet(os.path.join(HERE, "ref/factor_keys.parquet"))
H0 = pd.DataFrame({"date": k["key_date"], "instrument": "ID" + k["instrument_id"].astype(int).astype(str).str.zfill(4),
                   "factor": k["factor"]})
TOL = {"ic_mean": -0.0002, "ic_ir": -0.005, "ls_sharpe": -0.3, "stress": -0.005}


def basis(panel):
    B = panel[le.NEUTRAL_COLS].to_numpy(np.float64)
    ev = panel["is_eval"].to_numpy() & np.isfinite(B).all(axis=1)
    idx = np.flatnonzero(ev)
    keys = panel["key_date"].to_numpy()[idx]
    o = np.argsort(keys, kind="stable")
    idx, keys = idx[o], keys[o]
    out = []
    for ii in np.split(idx, np.flatnonzero(keys[1:] != keys[:-1]) + 1):
        if len(ii) < 50:
            continue
        Q, _ = np.linalg.qr(np.column_stack([np.ones(len(ii)), B[ii]]))
        out.append((ii, Q))
    return out


def resid(v, bs, n):
    v = np.nan_to_num(np.asarray(v, np.float64))
    out = np.full(n, np.nan)
    for ii, Q in bs:
        x = v[ii]
        if not x.std() > 0:
            continue
        out[ii] = x - Q @ (Q.T @ x)
    return out


def ev(v, panel):
    r = le.evaluate(pd.Series(v, index=panel.index), panel, neutralize=False)
    return {c: r[c] for c in ("ic_mean", "ic_ir", "ls_sharpe", "stress", "turnover")}


rows = []
for y in ("2023", "2024"):
    panel = A.panel_for(f"{y}-01", f"{y}-12")
    n = len(panel)
    bs = basis(panel)

    def align(df):
        m = panel[["key_date", "instrument"]].merge(df.rename(columns={"date": "key_date"}), how="left",
                                                    on=["key_date", "instrument"])
        return m["factor"].to_numpy(np.float64)
    h = resid(align(H0), bs, n)
    base = ev(h, panel)
    rows.append({"year": y, "name": "490", **base})
    secs = panel["key_date"].to_numpy()
    for nm in names:
        x = align(pd.read_parquet(os.path.join(HERE, "out", f"{nm}.parquet")))
        xr = resid(x, bs, n)
        m1 = ev(xr, panel)
        s = 1.0 if m1["ic_mean"] >= 0 else -1.0
        ok = np.isfinite(xr) & np.isfinite(h)            # 第 2 级：逐截面 xr 对 h 回归取残差
        d = pd.DataFrame({"k": secs[ok], "x": xr[ok], "h": h[ok]})
        g = d.groupby("k")
        bx = (g.apply(lambda t: np.dot(t["x"] - t["x"].mean(), t["h"] - t["h"].mean()) / max(np.dot(t["h"] - t["h"].mean(), t["h"] - t["h"].mean()), 1e-12),
                      include_groups=False))
        rr = np.full(n, np.nan)
        rr[np.flatnonzero(ok)] = (d["x"] - d["k"].map(bx) * d["h"]).to_numpy()
        m2 = ev(s * rr, panel)
        corr = np.corrcoef(xr[ok], h[ok])[0, 1]
        row = {"year": y, "name": nm, "ic1": m1["ic_mean"], "icir1": m1["ic_ir"], "turn1": m1["turnover"],
               "res_ic": m2["ic_mean"], "res_icir": m2["ic_ir"], "corr_h": corr}
        for w in WS:                                   # 第 3 级（同一投影：resid(h_raw + s·w·x) 与 h、xr 的线性组合等价，
            hx = resid(align(H0) + s * w * np.nan_to_num(x), bs, n)   # 但 h 已标准化过，这里直接对原始值重算更稳）
            m3 = ev(hx, panel)
            dd = {c: m3[c] - base[c] for c in m3}
            good = all(dd[c] >= TOL[c] for c in TOL) and dd["turnover"] <= 0.0
            row.update({f"w{w:g}_{c[:4]}": round(v, 5) for c, v in dd.items()})
            row[f"w{w:g}_ok"] = good
        rows.append(row)
        print(y, nm, {k_: (round(v, 4) if isinstance(v, float) else v) for k_, v in row.items() if k_ not in ("year", "name")}, flush=True)
out = pd.DataFrame(rows)
tag = os.environ.get("TAG", "_".join(names[:1]) + f"_{len(names)}")
out.to_csv(os.path.join(HERE, f"eval_{tag}.csv"), index=False)
print("saved", f"eval_{tag}.csv")
