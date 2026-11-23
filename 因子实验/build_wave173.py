# -*- coding: utf-8 -*-
"""第一百七十三批：日内价位时段类新源（纯 pandas，无 SQL 改动）。
- 998/1000：日内最低价出现时段 c_lotime（833 臂 / 925 臂）
- 999：日内最高价出现时段 c_hitime（833 臂）
- 1001：竞价价相对开盘的失衡 c_aucgap（压力榜首 895 臂）
四个全部是新信息源。
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

HITIME = ('    _pivP = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
          '    _arrP = _pivP.to_numpy(dtype="float64")\n'
          '    _allnan = np.all(np.isnan(_arrP), axis=1)\n'
          '    _idxH = np.nanargmax(np.where(np.isnan(_arrP), -np.inf, _arrP), axis=1).astype("float64")\n'
          '    _idxH[_allnan] = np.nan\n'
          '    _fracH = _idxH / max(_arrP.shape[1] - 1, 1)\n'
          '    _serH = pd.Series(_fracH, index=pd.MultiIndex.from_tuples(_pivP.index))\n'
          '    _keyP = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_hitime"] = _serH.reindex(_keyP).to_numpy()\n')
LOTIME = ('    _pivL = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
          '    _arrL = _pivL.to_numpy(dtype="float64")\n'
          '    _allnanL = np.all(np.isnan(_arrL), axis=1)\n'
          '    _idxL = np.nanargmin(np.where(np.isnan(_arrL), np.inf, _arrL), axis=1).astype("float64")\n'
          '    _idxL[_allnanL] = np.nan\n'
          '    _fracL = _idxL / max(_arrL.shape[1] - 1, 1)\n'
          '    _serL = pd.Series(_fracL, index=pd.MultiIndex.from_tuples(_pivL.index))\n'
          '    _keyL2 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_lotime"] = _serL.reindex(_keyL2).to_numpy()\n')
AUCGAP = ('    _pivG = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
          '    _keyG = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    def _pxG(_h):\n'
          '        return _pivG[_h].reindex(_keyG).to_numpy() if _h in _pivG.columns else np.full(len(df), np.nan)\n'
          '    _paG, _poG = _pxG("092500"), _pxG("093000")\n'
          '    df["c_aucgap"] = -np.where((_paG > 0) & (_poG > 0), np.log(np.where(_paG > 0, _paG, 1.0) / np.where(_poG > 0, _poG, 1.0)), np.nan)\n')

A833 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K833 = '"c_body", "c_utime"]\n'
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K925 = 'c_upshtwres", "c_condliq"]\n'
A895 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K895 = '"c_utime", "c_gap"]\n'

save(998, "行业市值中性化时点重心加最低价时段", add(load(833), LOTIME, "c_lotime", "0.15", A833, K833))
save(999, "行业市值中性化时点重心加最高价时段", add(load(833), HITIME, "c_hitime", "0.15", A833, K833))
def add925(code, tag, w, base):
    t = sub(base, A925, code + A925)
    t = sub(t, K925, K925.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)
save(1000, "多空臂925加最低价时段", add925(LOTIME, "c_lotime", "0.15", load(925)))
save(1001, "压力榜首895加竞价失衡", add(load(895), AUCGAP, "c_aucgap", "0.15", A895, K895))
