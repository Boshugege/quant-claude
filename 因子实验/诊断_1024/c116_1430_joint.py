# c116（10-24）：14:30 专用的联合线性组合（交叉年份：2023 拟合→2024 检验，反之亦然），看 c115 候选能把 14:30 IC 推到多高。
import glob, numpy as np, pandas as pd
d = pd.read_parquet("/srv/media/jyc_scorer/incoming/e2eslot/c115_feats.parquet")
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"]))
def base(job):
    bf = pd.read_parquet(glob.glob(f"/home/user/jyc_scorer/jobs/{job}/input/*.parquet")[0]); bf["date"] = pd.to_datetime(bf["date"])
    bf = bf[bf["date"].dt.strftime("%H%M") == "1430"].copy()
    re_ = ~bf["instrument"].str.match(r"^ID\d{4}$"); bf.loc[re_, "instrument"] = bf.loc[re_, "instrument"].map(R2I)
    return bf.dropna(subset=["instrument"])
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"]).rename(columns={"y": "ylab"})
lab = lab[lab["date"].dt.strftime("%H%M") == "1430"]
x = lab.merge(base("1024-020416-70ad").rename(columns={"factor": "base"}), on=["date", "instrument"]).merge(d, on=["date", "instrument"], how="left")
e = pd.concat([pd.read_parquet(f"/srv/media/jyc_scorer/incoming/e2eslot/e2e_R22a_{y}.parquet") for y in (2023, 2024)])
e["date"] = pd.to_datetime(e["date"]); x = x.merge(e.rename(columns={"factor": "e2e"}), on=["date", "instrument"], how="left")
def zs(s):
    g = s.groupby(x["date"]); return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0)
x["rl"] = x["ylab"].groupby(x["date"]).rank(pct=True) - 0.5
x["rd"] = x["drift"].groupby(x["date"]).rank(pct=True) - 0.5
x["ra"] = x["auc"].groupby(x["date"]).rank(pct=True) - 0.5
for c in ["drift", "auc", "y"]:
    s = x.sort_values("date").groupby("instrument_id")
F = ["base", "f_y_20", "f_drift_20", "f_cvx_20", "f_auc_20", "f_drift_5", "f_y_5", "f_r_day", "f_hl_pos", "f_seg", "f_pos_day", "f_v_last", "f_r_pm", "f_r_am30"]
Z = pd.DataFrame({f: zs(x[f]) for f in F})
yr = x["date"].dt.year.to_numpy()
def ics(v, t="rl"):
    q = pd.DataFrame({"d": x["date"], "v": v, "r": x[t]})
    ic = q.groupby("d")[["v", "r"]].apply(lambda g: g["v"].corr(g["r"]))
    return ic[ic.index.year == 2023].mean(), ic[ic.index.year == 2024].mean()
print("底座       标签 %.4f/%.4f  漂移 %.4f/%.4f  竞价 %.4f/%.4f" % (*ics(Z["base"]), *ics(Z["base"], "rd"), *ics(Z["base"], "ra")))
print("端到端     标签 %.4f/%.4f  漂移 %.4f/%.4f  竞价 %.4f/%.4f" % (*ics(zs(x["e2e"])), *ics(zs(x["e2e"]), "rd"), *ics(zs(x["e2e"]), "ra")))
import sys
if len(sys.argv) > 1: pass
for sets in [["base", "f_y_20"], ["base", "f_y_20", "f_cvx_20"], ["base", "f_y_20", "f_cvx_20", "f_drift_20", "f_auc_20"],
             ["base", "f_y_20", "f_cvx_20", "f_r_day", "f_hl_pos"], F]:
    X = Z[sets].to_numpy()
    preds = np.zeros(len(x)); coefs = {}
    for tr, te in ((2023, 2024), (2024, 2023)):
        m = (yr == tr) & np.isfinite(x["rl"].to_numpy())
        b = np.linalg.lstsq(X[m], x["rl"].to_numpy()[m], rcond=None)[0]
        preds[yr == te] = X[yr == te] @ b
        coefs[tr] = b / b[0]
    a = ics(preds); dd = ics(preds, "rd"); aa = ics(preds, "ra")
    print("样本外 %-60s 标签 %.4f/%.4f  漂移 %.4f/%.4f  竞价 %.4f/%.4f" % ("+".join(s.replace("f_", "") for s in sets[1:]) or "base", *a, *dd, *aa))
    print("        系数（以底座=1，2023 拟合）：" + " ".join("%s %.2f" % (s.replace("f_", ""), c) for s, c in zip(sets, coefs[2023])))
# 端到端在漂移上的优势，有多少能被这些候选解释：端到端对 底座+全部候选 的残差 与漂移的 IC
X = Z[F].to_numpy(); ee = zs(x["e2e"]).to_numpy()
res = np.zeros(len(x))
for t in (2023, 2024):
    m = yr == t; b = np.linalg.lstsq(X[m], ee[m], rcond=None)[0]; res[m] = ee[m] - X[m] @ b
print("端到端对 底座+候选 的残差：标签 %.4f/%.4f  漂移 %.4f/%.4f  竞价 %.4f/%.4f" % (*ics(res), *ics(res, "rd"), *ics(res, "ra")))
