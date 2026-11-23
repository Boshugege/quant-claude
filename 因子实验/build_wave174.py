# -*- coding: utf-8 -*-
"""第一百七十四批：把日内价位时段/竞价类新源接到压力臂与 IC_IR 臂（纯 pandas，无 SQL 改动）。
- 1002：压力臂866 + 最低价时段 c_lotime
- 1003：压力臂866 + 最高价时段 c_hitime
- 1004：IC_IR臂847 + 最低价时段 c_lotime
- 1005：多空臂925 + 竞价失衡 c_aucgap
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

A = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'

save(1002, "压力臂866加最低价时段", add(load(866), LOTIME, "c_lotime", "0.15", A, '"c_utime", "c_gap"]\n'))
save(1003, "压力臂866加最高价时段", add(load(866), HITIME, "c_hitime", "0.15", A, '"c_utime", "c_gap"]\n'))
save(1004, "ICIR臂847加最低价时段", add(load(847), LOTIME, "c_lotime", "0.15", A, '"c_utime", "c_indmwrev"]\n'))
t925 = sub(load(925), A925, AUCGAP + A925)
t925 = sub(t925, 'c_upshtwres", "c_condliq"]\n', 'c_upshtwres", "c_condliq", "c_aucgap"]\n')
save(1005, "多空臂925加竞价失衡", blend_add(t925, "c_aucgap", "0.15"))
