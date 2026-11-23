# -*- coding: utf-8 -*-
"""第一百六十八批：953（IC 均值 0.0815 新高）与 925 支路的收尾网格。
- 967：953 + 隔夜跳空 0.10
- 968：953 + 首窗成交占比 0.15
- 969：953 + 官方 LIQUIDTY 条件反转 0.15
- 970：953 + 早盘动量反转 0.15
- 971：925 中性化 2.5x（探针）
- 972：902 + 官方 RESVOL 条件反转 0.30
- 973：935 的换手加权行业内上影线残差 0.15 -> 0.25
- 974：925 条件反转分量权重 1.0 -> 1.5
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

def cond(col, tag):
    return ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
            '    _qS = pd.qcut(pd.Series(df["%s"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
            '    df["%s"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n' % (col, tag))
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

b953 = load(953)   # 935 的 1.75x，keep 尾到 c_condsize
A953 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K953 = 'c_upshtwres", "c_condsize"]\n'
def add953(code, tag, w):
    t = sub(b953, A953, code + A953)
    t = sub(t, K953, K953.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)
save(967, "条件反转扩展中性化加换手加权行业上影残差与SIZE及跳空010", add953(GAP, "c_gap", "0.10"))
save(968, "条件反转扩展中性化加换手加权行业上影残差与SIZE及首窗成交", add953(OPEN30, "c_open30", "0.15"))
save(969, "条件反转扩展中性化加换手加权行业上影残差与SIZE及LIQUIDTY条件", add953(cond("s_LIQUIDTY", "c_condliq"), "c_condliq", "0.15"))
save(970, "条件反转扩展中性化加换手加权行业上影残差与SIZE及早盘动量", add953(EARLY30, "c_first30", "0.15"))

b925 = load(925)
save(971, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件二点五", sub(b925, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 2.5)'))
save(974, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件条件反转权1.5",
     sub(b925, '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.15, 0.15, 0.30]',
              '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.5, 0.15, 0.15, 0.30]'))

b902 = load(902)
A902 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
t972 = sub(b902, A902, cond("s_RESVOL", "c_condresvol") + A902)
t972 = sub(t972, 'c_upshtwres"]\n', 'c_upshtwres", "c_condresvol"]\n')
save(972, "条件反转扩展中性化加换手加权行业上影残差与RESVOL条件", blend_add(t972, "c_condresvol", "0.30"))

b935 = load(935)
save(973, "条件反转扩展中性化加换手加权行业上影残差025与SIZE条件",
     sub(b935, '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condsize"], [1.0, 0.15, 0.15, 0.15]',
              '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condsize"], [1.0, 0.15, 0.25, 0.15]'))
