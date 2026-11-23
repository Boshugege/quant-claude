# -*- coding: utf-8 -*-
"""复活 Y_sd5／X_n5d（10-19）：把因子代码里近 5 日（1200 分钟）窗口改成平台安全写法。
平台上这两根为零，只因跨日 w5 窗口里对单笔成交额（lats）的写法失效；deal_number 本身完整。

改法（只动 A 的 _root_impl、B6 的 _root_impl_b6、B7 的 _root_impl_b7 三处的分钟 SQL）：
  - w5 上的 SUM／STDDEV 换成按日分区的前缀和：Σ|r|（cabs）、Σ r·(lats − lats_prev)（cnd）、Σlats、Σlats²、个数；
  - 结果多取 15:00 一行（全天合计）；
  - 查询结果交给 _w5_safe 在 pandas 里拼 1200 分钟窗口，再去掉 15:00 行。
半剂量模式（--half）再把 A 里 Y_sd5／X_n5d 的权重 ±0.10 → ±0.05、B6／B7 的权重减半。
用法：python3 因子实验/patch_w5_safe.py <源文件> <输出文件> <编号> <标题> [--half]
"""
import io
import re
import sys

src, dst, num, title = sys.argv[1:5]
half = "--half" in sys.argv[5:]
s = io.open(src, encoding="utf-8").read()

HELPER = '''

def _w5_safe(df):
    """近 5 日（1200 分钟）窗口的平台安全写法（10-19，复活 Y_sd5／X_n5d）。
    SQL 只给按 (股票, 交易日) 分区的前缀和，并多取 15:00 一行作全天合计。这里拼成
      窗口(k, t) = 前缀(k, t) + 累计(k−1) − 累计(k−6) − 前缀(k−5, t)，
    k 为该股票的第 k 个交易日。每天 240 行（09:31–15:00），无缺分钟时与原 ROWS BETWEEN 1199 PRECEDING 写法相同。
    产出 a5d、nd5（由 cabs、cnd）与 sd5（由 sl、sl2、nl），最后去掉 15:00 行。"""
    import numpy as np
    import pandas as pd
    cols = [c for c in ("cabs", "cnd", "sl", "sl2", "nl") if c in df.columns]
    df = df.copy()
    df["hm"] = df["hm"].astype(str)
    P = df[cols].apply(pd.to_numeric, errors="coerce").astype("float64").fillna(0.0)
    tot = df.loc[df["hm"] == "1500", ["instrument_id", "d"]].copy()
    tot[cols] = P.loc[tot.index].to_numpy()
    tot = tot.drop_duplicates(["instrument_id", "d"]).sort_values(["instrument_id", "d"]).reset_index(drop=True)
    tot["k"] = tot.groupby("instrument_id").cumcount()
    C = tot.groupby("instrument_id")[cols].cumsum().to_numpy()
    ck = pd.DataFrame(C, columns=cols, index=pd.MultiIndex.from_arrays([tot["instrument_id"].to_numpy(), tot["k"].to_numpy()]))
    x = df[["instrument_id", "d"]].merge(tot[["instrument_id", "d", "k"]], on=["instrument_id", "d"], how="left")
    kk = x["k"].to_numpy(dtype="float64")
    ki = np.where(np.isfinite(kk), kk, -999).astype(np.int64)
    iid = df["instrument_id"].to_numpy()

    def at(shift):
        return ck.reindex(pd.MultiIndex.from_arrays([iid, ki - shift])).to_numpy()

    c1, c6 = at(1), at(6)
    c6 = np.where((ki == 5)[:, None], 0.0, c6)
    pk = P.copy()
    pk["instrument_id"], pk["k"], pk["hm"] = iid, ki, df["hm"].to_numpy()
    pk = pk[pk["k"] >= 0].drop_duplicates(["instrument_id", "k", "hm"]).set_index(["instrument_id", "k", "hm"])[cols]
    p5 = np.nan_to_num(pk.reindex(pd.MultiIndex.from_arrays([iid, ki - 5, df["hm"].to_numpy()])).to_numpy(), nan=0.0)
    Wv = P.to_numpy() + c1 - c6 - p5
    Wv[ki < 5] = np.nan
    w = pd.DataFrame(Wv, columns=cols, index=df.index)
    if "cabs" in cols:
        df["a5d"] = w["cabs"]
    if "cnd" in cols:
        df["nd5"] = w["cnd"]
    if "sl" in cols:
        n_, s1, s2 = w["nl"], w["sl"], w["sl2"]
        var = (s2 - s1 * s1 / n_) / (n_ - 1)
        df["sd5"] = np.sqrt(var.where((n_ > 1) & (var >= 0)))
    return df[df["hm"] != "1500"].reset_index(drop=True)
'''

SEL_OLD = "SELECT * FROM m3 WHERE hm IN ('1000', '1030', '1100', '1130', '1330', '1400', '1430')"
SEL_NEW = "SELECT * FROM m3 WHERE hm IN ('1000', '1030', '1100', '1130', '1330', '1400', '1430', '1500')"


def func_span(text, name):
    a = text.index("\ndef %s(" % name) + 1
    m = re.search(r"\ndef ", text[a + 4:])
    b = a + 4 + m.start() + 1 if m else len(text)
    return a, b


def patch_func(text, name, pairs):
    a, b = func_span(text, name)
    body = text[a:b]
    for old, new in pairs:
        assert body.count(old) == 1, (name, old[:60], body.count(old))
        body = body.replace(old, new)
    return text[:a] + body + text[b:]


Q_CACHE_OLD = '_cache["min"] = dai.query(sql_min, filters={"date": [q5, end_date]}, compression=True).df()'
Q_CACHE_NEW = '_cache["min"] = _w5_safe(dai.query(sql_min, filters={"date": [q5, end_date]}, compression=True).df())'
Q_OLD = 'df = dai.query(sql_min, filters={"date": [q5, end_date]}, compression=True).df()'
Q_NEW = 'df = _w5_safe(dai.query(sql_min, filters={"date": [q5, end_date]}, compression=True).df())'

s = patch_func(s, "_root_impl", [
    ("SUM(ABS(r)) OVER w5 AS a5d, SUM(r * (lats - lats_prev)) OVER w5 AS nd5, STDDEV_SAMP(lats) OVER w5 AS sd5,",
     "SUM(lats) OVER w AS sl, SUM(lats * lats) OVER w AS sl2, COUNT(lats) OVER w AS nl,"),
    (SEL_OLD, SEL_NEW), (Q_CACHE_OLD, Q_CACHE_NEW)])
s = patch_func(s, "_root_impl_b6", [
    ("SUM(ABS(r)) OVER w AS cabs, STDDEV_SAMP(lats) OVER w5 AS sd5",
     "SUM(ABS(r)) OVER w AS cabs, SUM(lats) OVER w AS sl, SUM(lats * lats) OVER w AS sl2, COUNT(lats) OVER w AS nl"),
    (SEL_OLD, SEL_NEW), (Q_OLD, Q_NEW)])
s = patch_func(s, "_root_impl_b7", [
    ("SUM(ABS(r)) OVER w AS cabs, SUM(ABS(r)) OVER w5 AS a5d, SUM(r * (lats - lats_prev)) OVER w5 AS nd5",
     "SUM(ABS(r)) OVER w AS cabs, SUM(r * (lats - lats_prev)) OVER w AS cnd"),
    (SEL_OLD, SEL_NEW), (Q_OLD, Q_NEW)])

if half:
    old_w = '"Y_sd5": +0.10, "X_n5d": -0.10'
    assert old_w in s
    a, b = func_span(s, "_main_a0" if "\ndef _main_a0(" in s else "_main_a")
    body = s[a:b].replace('    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": +0.10, "Y_sd5": +0.10, "X_n5d": -0.10,',
                          '    W = {"L_lab10": +0.20, "M_q_lin": +0.10, "N_dir": -0.10, "W_vdir": -0.10, "W_zero": +0.10, "Y_sd5": +0.05, "X_n5d": -0.05,')
    assert body != s[a:b], "A 的权重行没找到"
    s = s[:a] + body + s[b:]
    for old, new in (("(main_b6, 0.0085, 0.0, {}),", "(main_b6, 0.00425, 0.0, {}),"), ("(main_b7, -0.017, 0.0, {}),", "(main_b7, -0.0085, 0.0, {}),")):
        assert s.count(old) == 1, old
        s = s.replace(old, new)

# 标题：main 的 docstring 第一行
m = re.search(r'def main\(datasources, start_date, end_date\):\n    """【[^】]*】', s)
assert m
s = s[:m.start()] + 'def main(datasources, start_date, end_date):\n    """【%s：%s】' % (num, title) + s[m.end():]
assert "w5 AS sd5" not in s.split("def _root_impl_b1(")[0], "A 里还有 w5"
s = s.rstrip("\n") + "\n" + HELPER
io.open(dst, "w", encoding="utf-8").write(s)
print("写出", dst, "（半剂量）" if half else "（完全复活）")
