# -*- coding: utf-8 -*-
"""第一百七十八批：日内路径类新源（纯 pandas，无 SQL 改动）。
- 1018：833 臂 + 开盘 30 分钟收益 c_openmom
- 1019：925 臂 + 开盘 30 分钟收益 c_openmom
- 1020：833 臂 + 日内最大回撤路径 c_intradd
- 1021：925 臂 + 日内最大回撤路径 c_intradd
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

OPENMOM = ('    _pivO = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
           '    _keyO = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
           '    def _pxO(_h):\n'
           '        return _pivO[_h].reindex(_keyO).to_numpy() if _h in _pivO.columns else np.full(len(df), np.nan)\n'
           '    _p0O, _p1O = _pxO("093000"), _pxO("100000")\n'
           '    df["c_openmom"] = -np.where((_p0O > 0) & (_p1O > 0), np.log(np.where(_p1O > 0, _p1O, 1.0) / np.where(_p0O > 0, _p0O, 1.0)), np.nan)\n')
INTRADD = ('    _pivD = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
           '    _arrD = np.where(_pivD.to_numpy(dtype="float64") > 0, _pivD.to_numpy(dtype="float64"), np.nan)\n'
           '    _runmax = np.fmax.accumulate(np.where(np.isnan(_arrD), -np.inf, _arrD), axis=1)\n'
           '    _dd = np.where(np.isfinite(_runmax) & (_runmax > 0), _arrD / np.where(_runmax > 0, _runmax, np.nan) - 1.0, np.nan)\n'
           '    _mdd = np.nanmin(_dd, axis=1)\n'
           '    _keyD3 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
           '    df["c_intradd"] = pd.Series(_mdd, index=pd.MultiIndex.from_tuples(_pivD.index)).reindex(_keyD3).to_numpy()\n')

A = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'

save(1018, "行业市值中性化时点重心加开盘动量", add(load(833), OPENMOM, "c_openmom", "0.15", A, '"c_body", "c_utime"]\n'))
save(1020, "行业市值中性化时点重心加日内最大回撤", add(load(833), INTRADD, "c_intradd", "0.15", A, '"c_body", "c_utime"]\n'))
def add925(code, tag, w):
    t = sub(load(925), A925, code + A925)
    t = sub(t, 'c_upshtwres", "c_condliq"]\n', 'c_upshtwres", "c_condliq", "%s"]\n' % tag)
    return blend_add(t, tag, w)
save(1019, "多空臂925加开盘动量", add925(OPENMOM, "c_openmom", "0.15"))
save(1021, "多空臂925加日内最大回撤", add925(INTRADD, "c_intradd", "0.15"))
