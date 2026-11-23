# c127（10-24，诊断）：因子侧信号对标签的"偏 IC"（控制端到端 R22a 预测后），看端到端还缺多少一档委托结构／订单流信息。
import os, glob, json, numpy as np, pandas as pd
JOBS = "/home/user/jyc_scorer/jobs"; E = "/srv/media/jyc_scorer/incoming/e2eslot"
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str); _m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"]))
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"])
lab = lab[(lab["date"] >= "2023-01-01") & (lab["date"] < "2025-01-01") & (lab["date"].dt.hour * 100 + lab["date"].dt.minute != 930)]
e = pd.concat([pd.read_parquet(f"{E}/e2e_R22a_{y}.parquet") for y in (2023, 2024)]); e["date"] = pd.to_datetime(e["date"])
x = lab.merge(e.rename(columns={"factor": "e2e"}), on=["date", "instrument"])
codes, u = pd.factorize(x["date"]); G = len(u); yr = pd.DatetimeIndex(u).year
def rk(v):
    return (pd.Series(np.asarray(v, float)).groupby(codes).rank(pct=True) - 0.5).fillna(0.0).to_numpy()
def cz(v):
    v = np.asarray(v, float); m = np.bincount(codes, v, minlength=G) / np.bincount(codes, minlength=G); return v - m[codes]
ry, re_ = cz(rk(x["y"])), cz(rk(x["e2e"]))
def pc(a, b):
    r = np.bincount(codes, a * b, minlength=G) / np.sqrt(np.bincount(codes, a * a, minlength=G) * np.bincount(codes, b * b, minlength=G))
    return np.nanmean(r[yr == 2023]), np.nanmean(r[yr == 2024])
def resid(a, b):                                  # 逐截面 a 对 b 回归取残差
    k = np.bincount(codes, a * b, minlength=G) / np.bincount(codes, b * b, minlength=G); return a - k[codes] * b
idx = {}
for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
    try: n = json.load(open(sp, encoding="utf-8")).get("name")
    except Exception: continue
    d = os.path.dirname(sp)
    if n and os.path.exists(os.path.join(d, "factor_raw.parquet")) and (n not in idx or d > idx[n]): idx[n] = d
print("端到端 R22a 自身 IC（原始秩相关口径）2023 %.4f 2024 %.4f" % pc(re_, ry))
for nm in ["s20_ql_s1only", "s35_of_l10_rel", "s20_ql_end", "s16_l10_clv", "s30_td_s1", "s21_sz_l10_a5", "s31_td_qv1", "s5_e_dimb_d"]:
    if nm not in idx: continue
    a = pd.read_parquet(os.path.join(idx[nm], "factor_raw.parquet"), columns=["date", "instrument", "factor"], filters=[("date", ">=", pd.Timestamp("2023-01-01"))])
    a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].map(lambda s: R2I.get(s, s))
    v = x[["date", "instrument"]].merge(a.drop_duplicates(["date", "instrument"]), on=["date", "instrument"], how="left")["factor"]
    f = cz(rk(v))
    print("%-16s IC %+.4f/%+.4f  与端到端相关 %+.3f  控制端到端后偏 IC %+.4f/%+.4f" % (nm, *pc(f, ry), pc(f, re_)[0], *pc(resid(f, re_), resid(ry, re_))))
