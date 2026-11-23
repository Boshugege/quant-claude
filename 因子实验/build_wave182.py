# -*- coding: utf-8 -*-
"""第一百八十二批：围绕本轮最强新源「日内最高价出现时段 c_hitime」(999 五维全超)做变体与跨臂。
- 1034：925 臂 + 最高价时段 c_hitime 0.15（补上 925 侧）
- 1035：833 臂 + 最高价时段 0.30（权重扫）
- 1036：960 臂 + 最高价时段 0.15（多空榜首叠加）
- 1037：833 臂 + 最高价时段 0.15 + 成交集中度 0.15（与 IC_IR 族叠加）
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

HITIME = ('    _pivP = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
          '    _arrP = _pivP.to_numpy(dtype="float64")\n'
          '    _allnan = np.all(np.isnan(_arrP), axis=1)\n'
          '    _idxH = np.nanargmax(np.where(np.isnan(_arrP), -np.inf, _arrP), axis=1).astype("float64")\n'
          '    _idxH[_allnan] = np.nan\n'
          '    _fracH = _idxH / max(_arrP.shape[1] - 1, 1)\n'
          '    _serH = pd.Series(_fracH, index=pd.MultiIndex.from_tuples(_pivP.index))\n'
          '    _keyP = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_hitime"] = _serH.reindex(_keyP).to_numpy()\n')
HHISRC = ('    _pivH = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
          '    _shareH = _pivH.div(_pivH.sum(axis=1).replace(0.0, np.nan), axis=0)\n'
          '    _hhiH = (_shareH.pow(2)).sum(axis=1)\n'
          '    _keyH = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_amthhi"] = -_hhiH.reindex(_keyH).to_numpy()\n')

A833 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'

# 1034: 925 + c_hitime
t = sub(load(925), A925, HITIME + A925)
t = sub(t, 'c_upshtwres", "c_condliq"]\n', 'c_upshtwres", "c_condliq", "c_hitime"]\n')
save(1034, "多空臂925加最高价时段", blend_add(t, "c_hitime", "0.15"))

# 1035: 833 + c_hitime 0.30 —— 由 999 直接改权重
t = load(999)
t = sub(t, ', 0.45, 0.15])', ', 0.45, 0.30])')
save(1035, "行业市值中性化时点重心加最高价时段030", t)

# 1036: 960 + c_hitime
t = sub(load(960), A925, HITIME + A925)
t = sub(t, 'c_upshtwres", "c_condliq"]\n', 'c_upshtwres", "c_condliq", "c_hitime"]\n')
save(1036, "多空榜首960加最高价时段", blend_add(t, "c_hitime", "0.15"))

# 1037: 999 + c_amthhi 0.15
t = sub(load(999), A833, HHISRC + A833)
t = sub(t, '"c_body", "c_utime", "c_hitime"]\n', '"c_body", "c_utime", "c_hitime", "c_amthhi"]\n')
save(1037, "行业市值中性化时点重心加最高价时段与成交集中度", blend_add(t, "c_amthhi", "0.15"))
