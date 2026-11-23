# c117（10-24，只作诊断，端到端的输出不进任何因子代码）：14:30 上端到端对"底座 + c115 候选"的残差（仍对漂移有 IC 0.11），
# 与因子库每个五年信号在 14:30 的相关；相关高的信号指向因子缺的信息。同时给出该信号对漂移、竞价跳动的 IC。
import os, glob, json, numpy as np, pandas as pd
JOBS = "/home/user/jyc_scorer/jobs"
d = pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c115_feats.parquet")
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"]))
def tid(f):
    f["instrument"] = f["instrument"].map(lambda s: R2I.get(s, s))
    ev = globals().get("EV")
    return f[f["instrument"].isin(ev)] if ev else f[f["instrument"].str.startswith("ID")]
def at1430(f):
    f["date"] = pd.to_datetime(f["date"]); return f[(f["date"].dt.hour == 14) & (f["date"].dt.minute == 30)].copy()
bf = tid(at1430(pd.read_parquet(glob.glob(f"{JOBS}/1024-020416-70ad/input/*.parquet")[0])))
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"]).rename(columns={"y": "ylab"})
lab = lab[lab["date"].dt.strftime("%H%M") == "1430"]
EV = set(lab["instrument"].unique())
x = lab.merge(bf.rename(columns={"factor": "base"}), on=["date", "instrument"]).merge(d, on=["date", "instrument"], how="left")
e = pd.concat([pd.read_parquet(f"/srv/media/jyc_scorer/incoming/e2eslot/e2e_R22a_{y}.parquet") for y in (2023, 2024)])
x = x.merge(at1430(e).rename(columns={"factor": "e2e"}), on=["date", "instrument"], how="left")
def zs(s):
    g = s.groupby(x["date"]); return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
F = ["base", "f_y_20", "f_drift_20", "f_cvx_20", "f_auc_20", "f_drift_5", "f_y_5", "f_r_day", "f_hl_pos", "f_seg", "f_pos_day", "f_v_last", "f_r_pm", "f_r_am30"]
X = np.column_stack([zs(x[f]).to_numpy() for f in F]); ee = zs(x["e2e"]).to_numpy(); yr = x["date"].dt.year.to_numpy()
res = np.zeros(len(x))
for t in (2023, 2024):
    m = yr == t; b = np.linalg.lstsq(X[m], ee[m], rcond=None)[0]; res[m] = ee[m] - X[m] @ b
x["res"] = res
x["rd"] = x["drift"].groupby(x["date"]).rank(pct=True) - 0.5
x["ra"] = x["auc"].groupby(x["date"]).rank(pct=True) - 0.5
codes, _u = pd.factorize(x["date"]); G = len(_u)
def _c(v):
    v = np.nan_to_num(np.asarray(v, float)); m = np.bincount(codes, v, minlength=G) / np.bincount(codes, minlength=G)
    return v - m[codes]
def pc(a, b):                    # 逐截面 Pearson 的均值（向量化）
    a, b = _c(a), _c(b)
    return np.nanmean(np.bincount(codes, a * b, minlength=G) / np.sqrt(np.bincount(codes, a * a, minlength=G) * np.bincount(codes, b * b, minlength=G)))
idx = {}
for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
    try: n = json.load(open(sp, encoding="utf-8")).get("name")
    except Exception: continue
    dd = os.path.dirname(sp)
    if n and os.path.exists(os.path.join(dd, "factor_raw.parquet")) and (n not in idx or dd > idx[n]): idx[n] = dd
names = [l.split()[0] for l in open("/tmp/nc136/addon2_list.txt") if l.strip()]
rows = []
for k, nm in enumerate(names):
    if nm not in idx: continue
    a = pd.read_parquet(os.path.join(idx[nm], "factor_raw.parquet"), columns=["date", "instrument", "factor"],
                        filters=[("date", ">=", pd.Timestamp("2023-01-01"))])
    a = tid(at1430(a[(a["date"] >= "2023-01-01") & (a["date"] < "2025-01-01")])).drop_duplicates(["date", "instrument"], keep="last")
    v = x[["date", "instrument"]].merge(a, on=["date", "instrument"], how="left")["factor"].astype(float)
    z = zs(v.fillna(v.groupby(x["date"]).transform("mean")))
    rows.append((nm, pc(z, x["res"]), pc(z, x["rd"]), pc(z, x["ra"]), pc(z, x["base"])))
    if k % 25 == 0: print(k, nm, flush=True)
R = pd.DataFrame(rows, columns=["job", "corr_res", "ic_drift", "ic_auc", "corr_base"])
R["a"] = R["corr_res"].abs()
R.sort_values("a", ascending=False).to_csv("/tmp/nc136/c117.csv", index=False)
print(R.sort_values("a", ascending=False).head(25).to_string(index=False, float_format=lambda v: "%+.3f" % v))
