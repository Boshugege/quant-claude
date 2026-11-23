# -*- coding: utf-8 -*-
"""第一百五十六批：在 833（IC_IR 榜首）管线上开新信息源，并做中性化诊断。
- 871：下影线占比（与上影线对称的新轴，需在 SQL 补 sum_dnsh）
- 872：尾盘 30 分钟动量反转（15:00/14:30 窗价）
- 873：早盘 30 分钟动量反转（10:00/09:30 窗价）
- 874：首窗成交占比（09:30 窗成交额/全天）
- 875：833 + 隔夜跳空 0.10（换手细扫，介于无跳空与 0.15 之间）
- 876：833 中性化仅用 4 特征（去掉 c_turn/c_mcap2/行业哑变量）——诊断扩展中性化对 IC_IR 的贡献
- 877：833 + 官方 RESVOL 前一日五分位内条件反转（新源）
- 878：833 + 尾盘动量反转 0.30（872 的强权重对冲）
"""
import io, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")

def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)

def save(num, name, text):
    path = os.path.join(D, "%d_%s.py" % (num, name))
    io.open(path, "w", encoding="utf-8").write(text)
    print("wrote", os.path.basename(path))

def sub(text, old, new):
    assert old in text, old[:90]
    return text.replace(old, new, 1)

b833 = load(833)
ANCHOR = "    df = df.replace([np.inf, -np.inf], np.nan)\n    keep = "
OLD_BLEND = 'result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45])'
KEEP_TAIL = '"c_body", "c_utime"]\n'

def add_axis(text, code, tag, weight):
    t = sub(text, ANCHOR, code + ANCHOR)
    t = sub(t, KEEP_TAIL, '"c_body", "c_utime", "%s"]\n' % tag)
    t = sub(t, OLD_BLEND, OLD_BLEND.replace('"c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45]',
                                            '"c_utime", "%s"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, %s]' % (tag, weight)))
    return t

# ---- 871：下影线占比（SQL 补列） ----
t = b833
t = sub(t, "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,",
           "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,\n"
           "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN LEAST(px_open, px_close) - px_low ELSE 0 END AS dnsh_c,")
t = sub(t, "            SUM(upsh_c) AS sum_upsh,", "            SUM(upsh_c) AS sum_upsh,\n            SUM(dnsh_c) AS sum_dnsh,")
t = sub(t, "f.sum_rng, f.sum_upsh, f.sum_deals, f.sum_upsh_t", "f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_deals, f.sum_upsh_t")
t = sub(t, '"sum_rng", "sum_upsh", "sum_deals", "sum_upsh_t"]', '"sum_rng", "sum_upsh", "sum_dnsh", "sum_deals", "sum_upsh_t"]')
t = add_axis(t, '    _dnsh = df["sum_dnsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n'
                '    df["c_dnsh"] = -roll(_dnsh, 40, "mean", 10)\n', "c_dnsh", "0.15")
save(871, "行业市值中性化时点重心加下影线占比", t)

# ---- 872/878：尾盘动量 ----
LATE_CODE = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
             '    _pivW = df.assign(_h=_hmsW).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
             '    _keyW = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
             '    _p15 = _pivW["150000"].reindex(_keyW).to_numpy() if "150000" in _pivW.columns else np.full(len(df), np.nan)\n'
             '    _p14 = _pivW["143000"].reindex(_keyW).to_numpy() if "143000" in _pivW.columns else np.full(len(df), np.nan)\n'
             '    _okW = (_p15 > 0) & (_p14 > 0)\n'
             '    df["c_last30"] = -np.where(_okW, np.log(np.where(_okW, _p15, 1.0) / np.where(_okW, _p14, 1.0)), np.nan)\n')
save(872, "行业市值中性化时点重心加尾盘动量反转", add_axis(b833, LATE_CODE, "c_last30", "0.15"))
save(878, "行业市值中性化时点重心加尾盘动量反转030", add_axis(b833, LATE_CODE, "c_last30", "0.30"))

# ---- 873：早盘动量 ----
EARLY_CODE = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
              '    _pivW = df.assign(_h=_hmsW).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
              '    _keyW = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
              '    _p10 = _pivW["100000"].reindex(_keyW).to_numpy() if "100000" in _pivW.columns else np.full(len(df), np.nan)\n'
              '    _p09 = _pivW["093000"].reindex(_keyW).to_numpy() if "093000" in _pivW.columns else np.full(len(df), np.nan)\n'
              '    _okW = (_p10 > 0) & (_p09 > 0)\n'
              '    df["c_first30"] = -np.where(_okW, np.log(np.where(_okW, _p10, 1.0) / np.where(_okW, _p09, 1.0)), np.nan)\n')
save(873, "行业市值中性化时点重心加早盘动量反转", add_axis(b833, EARLY_CODE, "c_first30", "0.15"))

# ---- 874：首窗成交占比 ----
OPEN_CODE = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
             '    _daW = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
             '    _o1W = df["sum_amt"].where(_hmsW == "093000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
             '    _osW = _o1W / _daW.where(_daW > 0)\n'
             '    df["c_open30"] = -roll(_osW, 40, "mean", 10)\n')
save(874, "行业市值中性化时点重心加首窗成交占比", add_axis(b833, OPEN_CODE, "c_open30", "0.15"))

# ---- 875：833 + 跳空 0.10 ----
b842 = load(842)
OLD_G = 'result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.15])'
save(875, "行业市值中性化时点重心045跳空010", sub(b842, OLD_G, OLD_G.replace("0.45, 0.15])", "0.45, 0.10])")))

# ---- 876：中性化诊断（仅 4 特征） ----
b876 = sub(b833, 'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"] + sorted([c for c in df.columns if c.startswith("i_")])',
                'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]')
save(876, "行业市值中性化时点重心四特征诊断", b876)

# ---- 877：RESVOL 条件反转 ----
COND = ('    # ===== 新轴：按前一日官方风格暴露分组的条件反转（五分位内去均值） =====\n'
        '    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
        '    _qS = pd.qcut(pd.Series(df["s_RESVOL"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
        '    df["c_condresvol"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n\n')
save(877, "行业市值中性化时点重心加风格RESVOL条件反转", add_axis(b833, COND, "c_condresvol", "0.30"))
