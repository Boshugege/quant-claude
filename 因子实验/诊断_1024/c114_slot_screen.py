# c114（10-24）：分时段附加项的离线快筛（不建评分机任务、不占盘）。
# 底座 = 1663 组合的不加缓冲带输出（任务 w14_ad_q30）；对 addon2_list.txt 的每个五年信号，在每个时段单独加 ±0.2·z(信号)，
# 算该时段逐截面 Pearson(组合, 标签居中秩) 的均值相对底座的变化（2023、2024 分开）。只用于排序，入围的再走 c105i（带缓冲带、评分机口径）。
import os, sys, json, glob, numpy as np, pandas as pd
JOBS = "/home/user/jyc_scorer/jobs"
BASE = sys.argv[1] if len(sys.argv) > 1 else "1024-020416-70ad"
W = [0.2, -0.2]
_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str)
_m = _m[_m["ok"].str.lower() == "true"]
R2I = dict(zip(_m["instrument"], _m["local_id"]))

idx = {}
for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
    try:
        n = json.load(open(sp, encoding="utf-8")).get("name")
    except Exception:
        continue
    d = os.path.dirname(sp)
    if n and os.path.exists(os.path.join(d, "factor_raw.parquet")) and (n not in idx or d > idx[n]):
        idx[n] = d

base = pd.read_parquet(glob.glob(os.path.join(JOBS, BASE, "input", "*.parquet"))[0])
base["date"] = pd.to_datetime(base["date"])
base = base[base["date"].dt.strftime("%H%M") != "0930"].reset_index(drop=True)
lab = pd.read_parquet("/home/user/jyc_scorer/data/labels_y_off.parquet", columns=["date", "instrument", "y"])
lab = lab[(lab["date"] >= "2023-01-01") & (lab["date"] < "2025-01-01")]
iid = base["instrument"].where(base["instrument"].str.match(r"^ID\d{4}$"), base["instrument"].map(R2I))
y = pd.DataFrame({"date": base["date"], "instrument": iid}).merge(lab, on=["date", "instrument"], how="left")["y"].to_numpy()
ok = np.isfinite(y)
sec = base["date"].to_numpy()
codes, uniq = pd.factorize(sec)
uniq = pd.DatetimeIndex(uniq)
slot = uniq.hour * 100 + uniq.minute
year = uniq.year
G = len(uniq)
# 标签居中秩（只在评估股票上）
ys = pd.Series(np.where(ok, y, np.nan))
r = (ys.groupby(codes).rank(pct=True) - 0.5).fillna(0.0).to_numpy()
okf = ok.astype(float)
n = np.bincount(codes, okf, minlength=G)

def zsec(x):
    s = pd.Series(x)
    g = s.groupby(codes)
    return ((s - g.transform("mean")) / g.transform("std").replace(0, np.nan)).fillna(0.0).to_numpy()

zb = zsec(base["factor"].to_numpy(float))
def cen(v):                       # 评估股票上逐截面去均值
    m = np.bincount(codes, v * okf, minlength=G) / np.maximum(n, 1)
    return (v - m[codes]) * okf
zbc = cen(zb)
Srr = np.bincount(codes, r * r * okf, minlength=G)
Sbr = np.bincount(codes, zbc * r, minlength=G)
Sbb = np.bincount(codes, zbc * zbc, minlength=G)
ic0 = Sbr / np.sqrt(Sbb * Srr)
SL = [1000, 1030, 1100, 1130, 1330, 1400, 1430]
def summ(ic):
    return {(s, yr): np.nanmean(ic[(slot == s) & (year == yr)]) for s in SL for yr in (2023, 2024)}
b0 = summ(ic0)
print("底座各时段 IC（2023／2024）：" + "  ".join("%d %.4f/%.4f" % (s, b0[(s, 2023)], b0[(s, 2024)]) for s in SL), flush=True)

names = [l.split()[0] for l in open("/tmp/nc136/addon2_list.txt") if l.strip()]
names = [x for x in names if not x.startswith(("s37_", "s40_"))] + [x for x in sys.argv[2:]]
rows = []
for k, nm in enumerate(names):
    if nm not in idx:
        print("缺", nm, flush=True); continue
    a = pd.read_parquet(os.path.join(idx[nm], "factor_raw.parquet"), columns=["date", "instrument", "factor"])
    a["date"] = pd.to_datetime(a["date"])
    a = a[(a["date"] >= "2023-01-01") & (a["date"] < "2025-01-01")].drop_duplicates(["date", "instrument"], keep="last")
    x = base[["date", "instrument"]].merge(a, on=["date", "instrument"], how="left")["factor"].astype(float).fillna(0.0).to_numpy()
    zs = cen(zsec(x))
    Ssr = np.bincount(codes, zs * r, minlength=G)
    Sss = np.bincount(codes, zs * zs, minlength=G)
    Sbs = np.bincount(codes, zbc * zs, minlength=G)
    for w in W:
        ic = (Sbr + w * Ssr) / np.sqrt((Sbb + w * w * Sss + 2 * w * Sbs) * Srr)
        sm = summ(ic)
        for s in SL:
            rows.append((nm, w, s, sm[(s, 2023)] - b0[(s, 2023)], sm[(s, 2024)] - b0[(s, 2024)]))
    if k % 25 == 0:
        print(k, nm, flush=True)
R = pd.DataFrame(rows, columns=["job", "w", "slot", "d23", "d24"])
R["dmin"] = R[["d23", "d24"]].min(axis=1)
R.to_csv("/tmp/nc136/c114_slot_screen.csv", index=False)
for s in SL:
    t = R[R["slot"] == s].sort_values("dmin", ascending=False).head(8)
    print("== %d" % s)
    for _, q in t.iterrows():
        print("  %-28s %+.1f  Δ23 %+.4f  Δ24 %+.4f" % (q["job"], q["w"], q["d23"], q["d24"]))
