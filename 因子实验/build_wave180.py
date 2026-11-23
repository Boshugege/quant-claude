# -*- coding: utf-8 -*-
"""第一百八十批：把本轮不伤臂的新源（成交集中度 c_amthhi、最低价时段 c_lotime）
叠加到 多空榜首 960 与 IC 均值榜首 953 上。
- 1026：960 + 成交集中度（目标多空）
- 1027：953 + 成交集中度（目标 IC 均值）
- 1028：960 + 最低价时段
- 1029：953 + 最低价时段
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

A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'

HHI = ('    _pivH = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
       '    _shareH = _pivH.div(_pivH.sum(axis=1).replace(0.0, np.nan), axis=0)\n'
       '    _hhiH = (_shareH.pow(2)).sum(axis=1)\n'
       '    _keyH = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_amthhi"] = -_hhiH.reindex(_keyH).to_numpy()\n')
LOTIME = ('    _pivL = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
          '    _arrL = _pivL.to_numpy(dtype="float64")\n'
          '    _allnanL = np.all(np.isnan(_arrL), axis=1)\n'
          '    _idxL = np.nanargmin(np.where(np.isnan(_arrL), np.inf, _arrL), axis=1).astype("float64")\n'
          '    _idxL[_allnanL] = np.nan\n'
          '    _fracL = _idxL / max(_arrL.shape[1] - 1, 1)\n'
          '    _serL = pd.Series(_fracL, index=pd.MultiIndex.from_tuples(_pivL.index))\n'
          '    _keyL2 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_lotime"] = _serL.reindex(_keyL2).to_numpy()\n')

def add(base_num, keep_tail, code, tag, w):
    t = sub(load(base_num), A925, code + A925)
    t = sub(t, keep_tail, keep_tail.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)

save(1026, "多空榜首960加成交集中度", add(960, 'c_upshtwres", "c_condliq"]\n', HHI, "c_amthhi", "0.15"))
save(1027, "IC均值榜首953加成交集中度", add(953, 'c_upshtwres", "c_condsize"]\n', HHI, "c_amthhi", "0.15"))
save(1028, "多空榜首960加最低价时段", add(960, 'c_upshtwres", "c_condliq"]\n', LOTIME, "c_lotime", "0.15"))
save(1029, "IC均值榜首953加最低价时段", add(953, 'c_upshtwres", "c_condsize"]\n', LOTIME, "c_lotime", "0.15"))
