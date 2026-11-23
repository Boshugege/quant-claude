import numpy as np, pandas as pd
from c1_upgrades import G, OUT
from g_go import inst_codes
c = G.c; z930 = G.z930
kd = pd.to_datetime(c.key_date); m = (kd.year == 2024)
h2 = np.load(OUT / "h2_c2pf_2024.npy"); h2[z930] = 0.0
ref = pd.DataFrame({"date": kd[m], "instrument": inst_codes(c)[np.flatnonzero(m)], "ref": h2[m]})
a = pd.read_parquet("/home/user/jyc_scorer/jobs/1018-011751-6b53/factor_raw.parquet")
a["date"] = pd.to_datetime(a["date"]); a["instrument"] = a["instrument"].astype(str)
x = ref.merge(a, on=["date", "instrument"]); x = x[x["date"].dt.strftime("%H%M") != "0930"]
d = x["factor"] - x["ref"]
print("带符号差 均值 %.4f 标准差 %.4f；ref 标准差 %.4f factor 标准差 %.4f" % (d.mean(), d.std(), x["ref"].std(), x["factor"].std()))
b = np.polyfit(x["ref"], x["factor"], 1); res = x["factor"] - np.polyval(b, x["ref"])
print("线性拟合 factor ≈ %.4f·ref + %.4f，残差标准差 %.4f（占 factor 标准差 %.1f%%）" % (b[0], b[1], res.std(), 100 * res.std() / x["factor"].std()))
rc = x.groupby("date")[["ref", "factor"]].corr(method="spearman").xs("ref", level=1)["factor"]
print("秩相关 < 0.98 的截面数 %d / %d" % ((rc < 0.98).sum(), len(rc)))
print(rc.sort_values().head(8).to_string())
print("按时段的秩相关均值", rc.groupby(rc.index.strftime("%H%M")).mean().round(4).to_dict())
