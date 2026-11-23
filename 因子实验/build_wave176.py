# -*- coding: utf-8 -*-
"""第一百七十六批：隔夜跳空波动率与日内价量相关等新源（纯 pandas，无 SQL 改动）。
- 1010：隔夜跳空波动率 c_gapvol（833 臂）
- 1011：隔夜跳空波动率 c_gapvol（压力臂 866）
- 1012：日内价量相关 c_pvcorr（833 臂）
- 1013：日内价量相关 c_pvcorr（925 臂）
全部新信息源。
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

def add(text, code, tag, w, anchor, keep_tail):
    t = sub(text, anchor, code + anchor)
    t = sub(t, keep_tail, keep_tail.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)

GAPVOL = ('    _hmsX = df["date"].dt.strftime("%H%M%S")\n'
          '    _day_open = df["last_px"].where(_hmsX == "093000").groupby([df["instrument"], df["day"]], sort=False).transform("max")\n'
          '    _gapx = np.where((_day_open > 0) & (df["pre_close"] > 0), np.log(np.where(_day_open > 0, _day_open, 1.0) / np.where(df["pre_close"] > 0, df["pre_close"], 1.0)), np.nan)\n'
          '    _dfg = pd.DataFrame({"i": df["instrument"].values, "d": df["day"].values, "g": _gapx}).dropna().drop_duplicates(["i", "d"]).sort_values(["i", "d"])\n'
          '    _dfg["gv"] = _dfg.groupby("i")["g"].transform(lambda s: s.rolling(40, min_periods=10).std())\n'
          '    _gvser = pd.Series(_dfg["gv"].values, index=pd.MultiIndex.from_arrays([_dfg["i"].values, _dfg["d"].values]))\n'
          '    _keyG = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_gapvol"] = -_gvser.reindex(_keyG).to_numpy()\n')
PVCORR = ('    _pivPx = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
          '    _pivAm = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
          '    _pxA = np.where(_pivPx.to_numpy(dtype="float64") > 0, _pivPx.to_numpy(dtype="float64"), np.nan)\n'
          '    _rrA = np.abs(np.diff(np.log(_pxA), axis=1))\n'
          '    _pivAm2 = _pivAm.reindex(index=_pivPx.index, columns=_pivPx.columns)\n'
          '    _amA = _pivAm2.to_numpy(dtype="float64")[:, 1:]\n'
          '    _mx = np.nanmean(_rrA, axis=1, keepdims=True); _my = np.nanmean(_amA, axis=1, keepdims=True)\n'
          '    _cov = np.nanmean((_rrA - _mx) * (_amA - _my), axis=1)\n'
          '    _sx = np.nanstd(_rrA, axis=1); _sy = np.nanstd(_amA, axis=1)\n'
          '    _pv = np.where((_sx > 0) & (_sy > 0), _cov / np.where((_sx * _sy) > 0, _sx * _sy, 1.0), np.nan)\n'
          '    _keyPv = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_pvcorr"] = -pd.Series(_pv, index=pd.MultiIndex.from_tuples(_pivPx.index)).reindex(_keyPv).to_numpy()\n')

A = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'

save(1010, "行业市值中性化时点重心加隔夜跳空波动", add(load(833), GAPVOL, "c_gapvol", "0.15", A, '"c_body", "c_utime"]\n'))
save(1011, "压力臂866加隔夜跳空波动", add(load(866), GAPVOL, "c_gapvol", "0.15", A, '"c_utime", "c_gap"]\n'))
save(1012, "行业市值中性化时点重心加价量相关", add(load(833), PVCORR, "c_pvcorr", "0.15", A, '"c_body", "c_utime"]\n'))
t925 = sub(load(925), A925, PVCORR + A925)
t925 = sub(t925, 'c_upshtwres", "c_condliq"]\n', 'c_upshtwres", "c_condliq", "c_pvcorr"]\n')
save(1013, "多空臂925加价量相关", blend_add(t925, "c_pvcorr", "0.15"))
