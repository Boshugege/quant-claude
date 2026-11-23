# -*- coding: utf-8 -*-
"""第一百八十一批：把「成交额日内分布/集中度」族里已验证抬 IC_IR/压力的几个源两两叠加（833 臂）。
991（HHI）已到 IC_IR 1.4996（本地榜首），987（首尾集中度）压力 1.3902，984（集合竞价）IC_IR 1.4940。
- 1030：833 + 集合竞价占比 + 成交集中度
- 1031：833 + 成交集中度 + 首尾成交集中度
- 1032：833 + 成交集中度 + 日内成交自相关
- 1033：833 + 首尾成交集中度 + 成交集中度(反向权重扫 0.30)
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

HHI = ('    _pivH = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
       '    _shareH = _pivH.div(_pivH.sum(axis=1).replace(0.0, np.nan), axis=0)\n'
       '    _hhiH = (_shareH.pow(2)).sum(axis=1)\n'
       '    _keyH = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_amthhi"] = -_hhiH.reindex(_keyH).to_numpy()\n')
EDGE = ('    _hmsE = df["date"].dt.strftime("%H%M%S")\n'
        '    _dE = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
        '    _edgeE = df["sum_amt"].where(_hmsE.isin(["093000", "150000"]), 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
        '    df["c_edge"] = -(_edgeE / _dE.where(_dE > 0))\n')
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

A = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'

def graft(base, tail, code, tag, w):
    t = sub(base, A, code + A)
    t = sub(t, tail, tail.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)

save(1030, "833加集合竞价占比与成交集中度", graft(load(984), '"c_body", "c_utime", "c_auction"]\n', HHI, "c_amthhi", "0.15"))
save(1031, "833加成交集中度与首尾集中度", graft(load(991), '"c_body", "c_utime", "c_amthhi"]\n', EDGE, "c_edge", "0.15"))
save(1032, "833加成交集中度与日内成交自相关", graft(load(991), '"c_body", "c_utime", "c_amthhi"]\n', AMT_AC, "c_amtac", "0.15"))
save(1033, "833加首尾集中度与成交集中度030", graft(load(987), '"c_body", "c_utime", "c_edge"]\n', HHI, "c_amthhi", "0.30"))
