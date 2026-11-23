# -*- coding: utf-8 -*-
"""第一百七十五批：日内路径形状类新源（纯 pandas，无 SQL 改动）。
- 1006/1009：日内收益已实现偏度 c_rskew（833 臂 / 925 臂）
- 1007：日内成交额分布的 realized vol c_amtvol（833 臂）
- 1008：尾盘30分钟收益 c_tailret（833 臂）
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

RSKEW = ('    _pivR = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
         '    _arrR = np.where(_pivR.to_numpy(dtype="float64") > 0, _pivR.to_numpy(dtype="float64"), np.nan)\n'
         '    _rR = np.diff(np.log(_arrR), axis=1)\n'
         '    _mrr = np.nanmean(_rR, axis=1, keepdims=True)\n'
         '    _srr = np.nanstd(_rR, axis=1)\n'
         '    _sk = np.nanmean((_rR - _mrr) ** 3, axis=1) / np.where(_srr ** 3 > 0, _srr ** 3, np.nan)\n'
         '    _sk[~np.isfinite(_sk)] = np.nan\n'
         '    _keyR = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
         '    df["c_rskew"] = -pd.Series(_sk, index=pd.MultiIndex.from_tuples(_pivR.index)).reindex(_keyR).to_numpy()\n')
AMTVOL = ('    _pivV = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
          '    _shareV = _pivV.div(_pivV.sum(axis=1).replace(0.0, np.nan), axis=0)\n'
          '    _vv = np.nanstd(_shareV.to_numpy(dtype="float64"), axis=1)\n'
          '    _keyV = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_amtvol"] = -pd.Series(_vv, index=pd.MultiIndex.from_tuples(_pivV.index)).reindex(_keyV).to_numpy()\n')
TAILRET = ('    _pivT = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
           '    _keyT = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
           '    def _pxT(_h):\n'
           '        return _pivT[_h].reindex(_keyT).to_numpy() if _h in _pivT.columns else np.full(len(df), np.nan)\n'
           '    _paT, _pbT = _pxT("143000"), _pxT("150000")\n'
           '    df["c_tailret"] = -np.where((_paT > 0) & (_pbT > 0), np.log(np.where(_paT > 0, _paT, 1.0) / np.where(_pbT > 0, _pbT, 1.0)), np.nan)\n')

A = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'

save(1006, "行业市值中性化时点重心加日内收益偏度", add(load(833), RSKEW, "c_rskew", "0.15", A, '"c_body", "c_utime"]\n'))
save(1007, "行业市值中性化时点重心加成交额波动", add(load(833), AMTVOL, "c_amtvol", "0.15", A, '"c_body", "c_utime"]\n'))
save(1008, "行业市值中性化时点重心加尾盘收益", add(load(833), TAILRET, "c_tailret", "0.15", A, '"c_body", "c_utime"]\n'))
t925 = sub(load(925), A925, RSKEW + A925)
t925 = sub(t925, 'c_upshtwres", "c_condliq"]\n', 'c_upshtwres", "c_condliq", "c_rskew"]\n')
save(1009, "多空臂925加日内收益偏度", blend_add(t925, "c_rskew", "0.15"))
