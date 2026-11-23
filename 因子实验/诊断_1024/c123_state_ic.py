# c123（10-24）：组合各成分的逐截面 IC（评分机口径，series.npz）是否随"截面时已知的市场状态"变化：
# 状态 = 同一天上一截面的标签离散度（窗口 (t−30, t] 在 t 时已结束）、上一截面该成分自己的 IC。
import os, glob, json, numpy as np, pandas as pd
JOBS = "/home/user/jyc_scorer/jobs"
idx = {}
for sp in glob.glob(os.path.join(JOBS, "*", "spec.json")):
    try: n = json.load(open(sp, encoding="utf-8")).get("name")
    except Exception: continue
    d = os.path.dirname(sp)
    if n and os.path.exists(os.path.join(d, "series.npz")) and (n not in idx or d > idx[n]): idx[n] = d
def ser(names):
    out = []
    for nm in names:
        z = np.load(os.path.join(idx[nm], "series.npz"))
        out.append(pd.DataFrame({k: z[k] for k in ("date", "day", "slot", "ic", "ysd")}))
    s = pd.concat(out).drop_duplicates("date").sort_values("date")
    s["date"] = pd.to_datetime(s["date"]); s["day"] = pd.to_datetime(s["day"])
    return s.set_index("date")
comps = {"1617": ["d_1617_2023", "d_1617_2024"], "of": ["s35_of_l10_rel"], "ql": ["s20_ql_s1only"], "qend": ["s20_ql_end"], "clv": ["s16_l10_clv"],
         "td": ["s30_td_s1"], "halfrev": ["s10_half_rev"], "rev2": ["s5_j_rev2"], "abnvol": ["s5_c_abnvol"], "amdrift": ["s10_am_drift"]}
S = {k: ser(v) for k, v in comps.items() if all(x in idx for x in v)}
base = S["1617"]
st = base[["day", "slot", "ysd"]].copy()
st["ysd_prev"] = st.groupby("day")["ysd"].shift(1)              # 同一天上一截面（窗口在本截面时刻结束）
st["q"] = np.floor(st.groupby("slot")["ysd_prev"].rank(pct=True) * 2.9999)
print("状态：上一截面标签离散度，按时段内三分位（低／中／高）；各成分本截面 IC 均值（2023–24，10:30–14:30）")
rows = []
for k, s in S.items():
    x = st.join(s[["ic"]], how="inner").dropna(subset=["q"])
    x = x[(x.index.year >= 2023)]
    g = x.groupby("q")["ic"].mean()
    ac = s["ic"].groupby(s["day"]).apply(lambda v: v.autocorr(1) if len(v) > 2 else np.nan).mean()
    rows.append((k, g.get(0, np.nan), g.get(1, np.nan), g.get(2, np.nan), x["ic"].corr(x["ysd_prev"]), ac, len(x)))
R = pd.DataFrame(rows, columns=["成分", "低", "中", "高", "IC~离散度相关", "日内IC自相关", "n"])
R["高−低"] = R["高"] - R["低"]
print(R.to_string(index=False, float_format=lambda v: "%+.4f" % v))
