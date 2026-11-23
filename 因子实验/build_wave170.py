# -*- coding: utf-8 -*-
"""第一百七十批：平台期后换新源——五个未用过的新轴，各投 IC_IR 臂（833）与多空/IC 均值臂（925）。
- 983/988：下影线时点重心（把 c_utime 的时点重心推广到「下影线发生时刻」，需补 SQL）
- 984/989：集合竞价成交占比（需从 09:25 竞价 bar 单独拆出成交额）
- 985/990：日内逐窗口成交额自相关（持续性 vs 突发性）
- 986：早盘收益 − 午后收益（日内动量的时段不对称）
- 987：首尾窗口成交集中度
"""
import io, os, re

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")

def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)

def save(num, name, text):
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write(text)
    print("wrote %d_%s.py" % (num, name))

def sub(t, old, new):
    assert old in t, old[:120]
    return t.replace(old, new, 1)

def blend_add(t, tag, w):
    m = re.search(r'(    result\["signal"\] = rank_blend\(\[)(.*?)(\], \[)(.*?)(\]\))', t)
    assert m, "no rank_blend"
    return t[:m.start()] + m.group(1) + m.group(2) + ', "%s"' % tag + m.group(3) + m.group(4) + ", %s" % w + m.group(5) + t[m.end():]

# ---------------- 新轴代码块（基于 df 层） ----------------
DN_TIME = ('    _gD = df.groupby(["instrument", "day"], sort=False)\n'
           '    _dcD = _gD["sum_dnsh_t"].transform("sum") / _gD["sum_dnsh"].transform("sum").where(_gD["sum_dnsh"].transform("sum") > 0)\n'
           '    df["c_dntime"] = -(_dcD - 5.0)\n')
AMT_AC = ('    _pivA = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
          '    _arrA = _pivA.to_numpy(dtype="float64")\n'
          '    if _arrA.shape[1] >= 3:\n'
          '        _xa, _ya = _arrA[:, 1:], _arrA[:, :-1]\n'
          '        _mxa, _mya = np.nanmean(_xa, axis=1, keepdims=True), np.nanmean(_ya, axis=1, keepdims=True)\n'
          '        _cova = np.nanmean((_xa - _mxa) * (_ya - _mya), axis=1)\n'
          '        _sxa, _sya = np.nanstd(_xa, axis=1), np.nanstd(_ya, axis=1)\n'
          '        _acA = np.where((_sxa > 0) & (_sya > 0), _cova / np.where((_sxa * _sya) > 0, _sxa * _sya, 1.0), np.nan)\n'
          '    else:\n'
          '        _acA = np.full(len(_pivA), np.nan)\n'
          '    _serA = pd.Series(_acA, index=pd.MultiIndex.from_tuples(_pivA.index))\n'
          '    _keyA = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_amtac"] = -_serA.reindex(_keyA).to_numpy()\n')
AM_PM = ('    _pivM = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
         '    _keyM = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
         '    def _pxM(_h):\n'
         '        return _pivM[_h].reindex(_keyM).to_numpy() if _h in _pivM.columns else np.full(len(df), np.nan)\n'
         '    _p09M, _p11M, _p13M, _p15M = _pxM("093000"), _pxM("113000"), _pxM("133000"), _pxM("150000")\n'
         '    _okAM = (_p11M > 0) & (_p09M > 0)\n'
         '    _okPM = (_p15M > 0) & (_p13M > 0)\n'
         '    _rAM = np.where(_okAM, np.log(np.where(_okAM, _p11M, 1.0) / np.where(_okAM, _p09M, 1.0)), np.nan)\n'
         '    _rPM = np.where(_okPM, np.log(np.where(_okPM, _p15M, 1.0) / np.where(_okPM, _p13M, 1.0)), np.nan)\n'
         '    df["c_amdif"] = -(_rAM - _rPM)\n')
EDGE = ('    _hmsE = df["date"].dt.strftime("%H%M%S")\n'
        '    _dE = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
        '    _edgeE = df["sum_amt"].where(_hmsE.isin(["093000", "150000"]), 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
        '    df["c_edge"] = -(_edgeE / _dE.where(_dE > 0))\n')

# 983/984 需要的 SQL 补列
SQL_DNSH_T_ADD = ('            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,\n'
                  '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN LEAST(px_open, px_close) - px_low ELSE 0 END AS dnsh_c,\n'
                  '            (CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN (LEAST(px_open, px_close) - px_low) * seg_i ELSE 0 END) AS dnsh_t_c,\n')
SQL_AUC_ADD = '            CASE WHEN hms = \'092500\' THEN amt_c ELSE 0 END AS auc_amt,\n'

UPSH_C_LINE = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,\n'
DNSH_C_LINE = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN LEAST(px_open, px_close) - px_low ELSE 0 END AS dnsh_c,\n'
UPSH_T_LINE = '            (CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN (px_high - GREATEST(px_open, px_close)) * seg_i ELSE 0 END) AS upsh_t_c,\n'
DNSH_T_LINE = '            (CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN (LEAST(px_open, px_close) - px_low) * seg_i ELSE 0 END) AS dnsh_t_c,\n'

def sql_dnsh_t(t):
    t = sub(t, UPSH_C_LINE, UPSH_C_LINE + DNSH_C_LINE)
    t = sub(t, UPSH_T_LINE, UPSH_T_LINE + DNSH_T_LINE)
    t = sub(t, "            SUM(upsh_c) AS sum_upsh,", "            SUM(upsh_c) AS sum_upsh,\n            SUM(dnsh_c) AS sum_dnsh,\n            SUM(dnsh_t_c) AS sum_dnsh_t,")
    return t

def sql_auc(t):
    t = sub(t, "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,", "            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n" + SQL_AUC_ADD)
    t = sub(t, "            SUM(upsh_c) AS sum_upsh,", "            SUM(upsh_c) AS sum_upsh,\n            SUM(auc_amt) AS sum_auc,")
    return t

def passthru(t, cols):
    for c in cols:
        t = sub(t, "f.sum_upsh,", "f.sum_upsh, f.%s," % c) if c not in t else t
    return t

# ---------- 833 臂（IC_IR 榜首支路） ----------
b833 = load(833)
A833 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K833 = '"c_body", "c_utime"]\n'
B833 = '["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45]'

def add833(text, code, tag, w, anchor=A833):
    t = sub(text, anchor, code + anchor)
    t = sub(t, K833, '"c_body", "c_utime", "%s"]\n' % tag)
    return blend_add(t, tag, w)

# 983：下影线时点重心
t = b833.replace('f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals, f.sum_upsh_t',
                 'f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_dnsh_t, f.sum_deals, f.sum_upsh_t')
t = sub(t, '"sum_rng", "sum_upsh", "sum_deals", "sum_upsh_t"]', '"sum_rng", "sum_upsh", "sum_dnsh", "sum_dnsh_t", "sum_deals", "sum_upsh_t"]')
t = sql_dnsh_t(t)
save(983, "行业市值中性化时点重心加下影线时点重心", add833(t, DN_TIME, "c_dntime", "0.15"))

# 984：集合竞价成交占比
t = b833.replace('f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals, f.sum_upsh_t',
                 'f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_auc, f.sum_deals, f.sum_upsh_t')
t = sub(t, '"sum_rng", "sum_upsh", "sum_deals", "sum_upsh_t"]', '"sum_rng", "sum_upsh", "sum_auc", "sum_deals", "sum_upsh_t"]')
t = sql_auc(t)
AUC = ('    _hmsU = df["date"].dt.strftime("%H%M%S")\n'
       '    _dU = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
       '    _aU = df["sum_auc"].groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
       '    df["c_auction"] = -(_aU / _dU.where(_dU > 0))\n')
save(984, "行业市值中性化时点重心加集合竞价占比", add833(t, AUC, "c_auction", "0.15"))

save(985, "行业市值中性化时点重心加日内成交自相关", add833(b833, AMT_AC, "c_amtac", "0.15"))
save(986, "行业市值中性化时点重心加早午收益差", add833(b833, AM_PM, "c_amdif", "0.15"))
save(987, "行业市值中性化时点重心加首尾成交集中度", add833(b833, EDGE, "c_edge", "0.15"))

# ---------- 925 臂（IC 均值/多空榜首支路） ----------
b925 = load(925)
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K925 = 'c_upshtwres", "c_condliq"]\n'
def add925(code, tag, w, base=None):
    t = sub(base if base is not None else b925, A925, code + A925)
    t = sub(t, K925, K925.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)

# 988：下影线时点重心（925 的 SQL 已含 sum_dnsh，但无 sum_dnsh_t / seg_i 乘子）
b925m = b925.replace('f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals, f.sum_upsh_t',
                     'f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_dnsh_t, f.sum_body, f.sum_deals, f.sum_upsh_t')
b925m = sub(b925m, '"sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals", "sum_upsh_t"]',
                 '"sum_rng", "sum_upsh", "sum_dnsh", "sum_dnsh_t", "sum_body", "sum_deals", "sum_upsh_t"]')
b925m = sub(b925m, UPSH_T_LINE, UPSH_T_LINE + DNSH_T_LINE)
b925m = sub(b925m, "            SUM(dnsh_c) AS sum_dnsh,", "            SUM(dnsh_c) AS sum_dnsh,\n            SUM(dnsh_t_c) AS sum_dnsh_t,")
save(988, "多空臂925加下影线时点重心", add925(DN_TIME, "c_dntime", "0.15", base=b925m))
save(989, "多空臂925加日内成交自相关", add925(AMT_AC, "c_amtac", "0.15"))
save(990, "多空臂925加早午收益差", add925(AM_PM, "c_amdif", "0.15"))
