# -*- coding: utf-8 -*-
"""第一百六十五批：围绕 925（IC 均值 0.0812 / 多空 48.09，双榜首）做参数与杠杆叠加。
- 943：925 中性化 2.25x
- 944：925 时点重心 0.30
- 945：925 + 隔夜跳空 0.10
- 946：925 + 官方 SIZE 条件反转 0.15
- 947：925 + 首窗成交占比 0.15
- 948：925 + 早盘动量反转 0.15
- 949：925 时点重心 0.45
- 950：902 + LIQUIDTY 条件反转 0.45
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
    assert old in t, old[:110]
    return t.replace(old, new, 1)

def blend_add(t, tag, w):
    m = re.search(r'(    result\["signal"\] = rank_blend\(\[)(.*?)(\], \[)(.*?)(\]\))', t)
    assert m, "no rank_blend"
    return t[:m.start()] + m.group(1) + m.group(2) + ', "%s"' % tag + m.group(3) + m.group(4) + ", %s" % w + m.group(5) + t[m.end():]

GAP = ('    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
OPEN30 = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
          '    _daW = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
          '    _o1W = df["sum_amt"].where(_hmsW == "093000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
          '    df["c_open30"] = -roll(_o1W / _daW.where(_daW > 0), 40, "mean", 10)\n')
EARLY30 = ('    _pivE = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
           '    _keyE = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
           '    _p10 = _pivE["100000"].reindex(_keyE).to_numpy() if "100000" in _pivE.columns else np.full(len(df), np.nan)\n'
           '    _p09 = _pivE["093000"].reindex(_keyE).to_numpy() if "093000" in _pivE.columns else np.full(len(df), np.nan)\n'
           '    _okE = (_p10 > 0) & (_p09 > 0)\n'
           '    df["c_first30"] = -np.where(_okE, np.log(np.where(_okE, _p10, 1.0) / np.where(_okE, _p09, 1.0)), np.nan)\n')
CONDSIZE = ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
            '    _qS = pd.qcut(pd.Series(df["s_SIZE"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
            '    df["c_condsize"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n')

b925 = load(925)
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K925 = 'c_upshtwres", "c_condliq"]\n'
save(943, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件二点二五", sub(b925, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 2.25)'))
save(944, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件与时点重心030",
     sub(b925, '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.15, 0.15, 0.30]',
              '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.30, 0.15, 0.30]'))
t945 = sub(b925, A925, GAP + A925)
t945 = sub(t945, K925, 'c_upshtwres", "c_condliq", "c_gap"]\n')
save(945, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件及跳空", blend_add(t945, "c_gap", "0.10"))
t946 = sub(b925, A925, CONDSIZE + A925)
t946 = sub(t946, K925, 'c_upshtwres", "c_condliq", "c_condsize"]\n')
save(946, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY及SIZE条件", blend_add(t946, "c_condsize", "0.15"))
t947 = sub(b925, A925, OPEN30 + A925)
t947 = sub(t947, K925, 'c_upshtwres", "c_condliq", "c_open30"]\n')
save(947, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件及首窗成交", blend_add(t947, "c_open30", "0.15"))
t948 = sub(b925, A925, EARLY30 + A925)
t948 = sub(t948, K925, 'c_upshtwres", "c_condliq", "c_first30"]\n')
save(948, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件及早盘动量", blend_add(t948, "c_first30", "0.15"))
save(949, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件与时点重心045",
     sub(b925, '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.15, 0.15, 0.30]',
              '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.45, 0.15, 0.30]'))
b902 = load(902)
save(950, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件045",
     blend_add(b902, "c_condliq", "0.45"))
