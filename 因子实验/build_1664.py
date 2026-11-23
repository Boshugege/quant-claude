# -*- coding: utf-8 -*-
"""1653 系列 + 缓冲带之前的风格剔除（10-24）。
评分机（与平台同口径）先把因子逐截面对 10 个官方风格 + 申万一级行业做 OLS 取残差，五项指标（含多空的腿、换手的腿）都按残差排名算；
我们的最小抬升缓冲带原来按原始合成分的排名定腿，与评分机的腿不一致，迟滞的效果被风格剔除打散。
这里在 build_1653 的输出上，于缓冲带之前插入同样的剔除（暴露取前一交易日，盘中截面拿不到当天的），让缓冲带直接作用在评分机看到的排名上。
本地（评分机 c105j，2023–24）：同一 β 换手大降，同一换手 IC、IC_IR、多空都升。
用法：python3 因子实验/build_1664.py <编号> <名字> <buf> <m> <附加项…> [<q_w>]"""
import subprocess
import sys

num, title = sys.argv[1], sys.argv[2]
subprocess.run([sys.executable, "因子实验/build_1653.py"] + sys.argv[1:], check=True)
DST = "因子代码/%s_%s.py" % (num, title)
s = open(DST, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep("""    if _POST_BUF > 0:
        d = d.sort_values(["date", "instrument"]).reset_index(drop=True)""",
    """    if _POST_RESID:                                                              # 缓冲带之前：逐截面对官方 10 风格 + 申万一级行业取残差（前一交易日暴露），腿与评分机一致
        d["z"] = _resid_styles(d, datasources, start_date, end_date)
        d.loc[d["date"].dt.strftime("%%H%%M") == "0930", "z"] = 0.0
        d["z"] = _z(d["z"])
    if _POST_BUF > 0:
        d = d.sort_values(["date", "instrument"]).reset_index(drop=True)""".replace("%%", "%"))
rep("_POST_EW, _POST_BUF, _POST_ANCHOR_W, _POST_Q_W, _POST_Q_MODE, _POST_Q_SM = ",
    "_POST_RESID = True                                                                  # 缓冲带之前做风格 + 行业剔除\n_POST_EW, _POST_BUF, _POST_ANCHOR_W, _POST_Q_W, _POST_Q_MODE, _POST_Q_SM = ")
HELPER = '''

_STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]


def _resid_styles(d, datasources, start_date, end_date):
    """d["z"] 逐截面对 10 个官方风格暴露 + 申万一级行业哑变量（含截距）做 OLS，返回残差（暴露缺失的股票原值不动）。
    暴露表第 t 天的值在当天收盘后才可得，一律取该股票前一交易日的值。09:30 截面跳过（之后置 0）。"""
    import numpy as np
    import pandas as pd
    import dai

    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")
    qs = (pd.to_datetime(start_date) - pd.Timedelta(days=20)).strftime("%Y-%m-%d %H:%M:%S")
    e = dai.query("SELECT date, instrument, %s, industry_level1_code FROM %s" % (", ".join(_STYLES), exposure_table),
                  filters={"date": [qs, end_date]}, compression=True).df()
    e["day"] = pd.to_datetime(e["date"]).dt.normalize()
    e["instrument"] = e["instrument"].astype(str)
    e = e.drop_duplicates(["instrument", "day"]).sort_values(["instrument", "day"]).reset_index(drop=True)
    ind = sorted(e["industry_level1_code"].dropna().astype(str).unique())
    X = pd.DataFrame({c: pd.to_numeric(e[c], errors="coerce").astype(float) for c in _STYLES})
    for c in ind:
        X["i_" + c] = (e["industry_level1_code"].astype(str) == c).astype(float)
    X[e["industry_level1_code"].isna().to_numpy()] = np.nan
    cols = list(X.columns)
    X["instrument"], X["day"] = e["instrument"].to_numpy(), e["day"].to_numpy()
    X[cols] = X.groupby("instrument")[cols].shift(1)                             # 前一交易日
    m = pd.DataFrame({"instrument": d["instrument"].to_numpy(), "day": d["date"].dt.normalize().to_numpy()}).merge(
        X, on=["instrument", "day"], how="left")
    A = m[cols].to_numpy(float)
    v = d["z"].to_numpy(float).copy()
    ok = np.isfinite(A).all(axis=1) & np.isfinite(v) & (d["date"].dt.strftime("%H%M") != "0930").to_numpy()
    out = v.copy()
    for _, idx in d.groupby("date", sort=False).indices.items():
        k = idx[ok[idx]]
        if len(k) < 50:
            continue
        Ak = np.column_stack([np.ones(len(k)), A[k]])
        b = np.linalg.lstsq(Ak, v[k], rcond=None)[0]
        out[k] = v[k] - Ak @ b
    return out
'''
k = s.index('\nif __name__ == "__main__":')
s = s[:k].rstrip("\n") + HELPER + "\n" + s[k:]
rep("腿边界缓冲带（最小抬升", "缓冲带之前对官方 10 风格 + 申万一级行业取残差（前一交易日暴露，与评分机的腿一致）；腿边界缓冲带（最小抬升")
open(DST, "w", encoding="utf-8").write(s)
print("加缓冲带前风格剔除", DST)
