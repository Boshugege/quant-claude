# c113（10-24）：14:30 标签拆成"连续竞价漂移"（14:57 收盘价 / 14:31–14:57 VWAP − 1）与"收盘竞价跳动"（15:00 / 14:57 − 1），
# 看因子组合 1663 与端到端 R22a 分别预测哪一部分（逐日截面秩相关的均值），2023–24。
import sys, numpy as np, pandas as pd
B = "/srv/media/jyc_scorer/bar1m_parquet"
J = "/home/user/jyc_scorer/jobs"
E = "/srv/media/jyc_scorer/incoming/e2eslot"
cols = ["date", "instrument_id", "close", "volume", "amount", "high", "low"]
parts = []
for y in (2023, 2024):
    for m in range(1, 13):
        p = pd.read_parquet(f"{B}/{y}{m:02d}.0.parquet", columns=cols)
        hm = p["date"].dt.hour * 100 + p["date"].dt.minute
        p = p[(hm > 1430) & (hm <= 1500)].copy()
        hm = p["date"].dt.hour * 100 + p["date"].dt.minute
        p["day"] = p["date"].dt.normalize()
        vw = p["amount"] / p["volume"].where(p["volume"] > 0)
        ok = vw.notna() & (vw >= p["low"] * 0.98) & (vw <= p["high"] * 1.02)
        p["amt"] = p["amount"].where(ok, 0.0); p["vol"] = p["volume"].where(ok, 0.0)
        p = p.sort_values(["instrument_id", "date"])
        cont = p[hm <= 1457]
        g_all = p.groupby(["instrument_id", "day"])
        g_c = cont.groupby(["instrument_id", "day"])
        auc = p[hm == 1500].set_index(["instrument_id", "day"])
        d = pd.DataFrame({"vwap": g_all["amt"].sum() / g_all["vol"].sum().where(lambda s: s > 0),
                          "c1500": g_all["close"].last(),
                          "vwap_c": g_c["amt"].sum() / g_c["vol"].sum().where(lambda s: s > 0),
                          "c1457": g_c["close"].last(),
                          "v_all": g_all["vol"].sum()})
        d["v_auc"] = auc["vol"].reindex(d.index)
        parts.append(d)
    print(y, flush=True)
d = pd.concat(parts).reset_index()
d["y"] = d["c1500"] / d["vwap"] - 1
d["auc"] = d["c1500"] / d["c1457"] - 1
d["drift"] = d["c1457"] / d["vwap_c"] - 1
d["ashare"] = d["v_auc"] / d["v_all"]
d["date"] = d["day"] + pd.Timedelta("14:30:00")
d["instrument"] = "ID" + d["instrument_id"].astype(int).astype(str).str.zfill(4)
d = d.replace([np.inf, -np.inf], np.nan)
print("拆分检验：y ≈ (1−竞价占比)·(漂移 + 竞价跳动)… 秩相关 y~auc %.3f  y~drift %.3f  竞价占比中位 %.3f" % (
    d[["y", "auc"]].corr("spearman").iloc[0, 1], d[["y", "drift"]].corr("spearman").iloc[0, 1], d["ashare"].median()))

_m = pd.read_csv("/home/user/jyc_scorer/data/exposure/id_map.csv", dtype=str)
R2I = dict(zip(_m.loc[_m["ok"].str.lower() == "true", "instrument"], _m.loc[_m["ok"].str.lower() == "true", "local_id"]))

def load_factor(path):
    f = pd.read_parquet(path)
    f["date"] = pd.to_datetime(f["date"])
    f = f[f["date"].dt.strftime("%H%M") == "1430"][["date", "instrument", "factor"]].copy()
    real = ~f["instrument"].str.match(r"^ID\d{4}$")           # 因子任务多用真实代码，按评分机的 id_map.csv 换成 ID0006 伪代码
    f.loc[real, "instrument"] = f.loc[real, "instrument"].map(R2I)
    return f.dropna(subset=["instrument"])

srcs = {"1663": load_factor(f"{J}/1024-004413-6340/factor_raw.parquet"),
        "e2e": pd.concat([load_factor(f"{E}/e2e_R22a_2023.parquet"), load_factor(f"{E}/e2e_R22a_2024.parquet")])}
for k in sys.argv[1:]:          # 额外的单项：名字=任务目录
    n, j = k.split("=")
    srcs[n] = load_factor(f"{J}/{j}/factor_raw.parquet")
for n, f in srcs.items():
    x = f.merge(d[["date", "instrument", "y", "auc", "drift"]], on=["date", "instrument"])
    out = []
    for t in ("y", "auc", "drift"):
        ic = x.groupby("date")[["factor", t]].apply(lambda g: g["factor"].corr(g[t], method="spearman"))
        out.append("%s %.4f(IR %.2f)" % (t, ic.mean(), ic.mean() / ic.std()))
        for yr in ("2023", "2024"):
            out.append("%s %.4f" % (yr, ic[ic.index.year == int(yr)].mean()))
    print("%-6s n=%d  " % (n, len(x)) + "  ".join(out), flush=True)
