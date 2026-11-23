# -*- coding: utf-8 -*-
"""第一百六十七批：925（IC 均值 0.0812 / 多空 48.09）支路上的风格条件轴扩张与权重网格。
- 959：925 + 换手加权行业内上影线残差 0.25
- 960：925 + 官方 LIQUIDTY 条件反转 0.15
- 961：925 + 官方 SIZE 条件反转 0.30
- 962：925 + 官方 BETA 条件反转 0.15
- 963：925 + 官方 MOMENTUM 条件反转 0.15
- 964：925 + 官方 RESVOL 条件反转 0.15
- 965：925 + 下影线占比 0.15
- 966：925 + 隔夜跳空 0.15
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

b925 = load(925)
A = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K = 'c_upshtwres", "c_condliq"]\n'
DNSH = '    _dnsh = df["sum_dnsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_dnsh"] = -roll(_dnsh, 40, "mean", 10)\n'
GAP = ('    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')

def add(code, tag, w):
    t = sub(b925, A, code + A)
    t = sub(t, K, K.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)

save(959, "条件反转扩展中性化加换手加权行业上影残差025与LIQUIDTY条件",
     sub(b925, '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.15, 0.15, 0.30]',
              '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.15, 0.25, 0.30]'))
save(960, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件015",
     sub(b925, '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.15, 0.15, 0.30]',
              '["c_cond_turnself", "c_utime", "c_upshtwres", "c_condliq"], [1.0, 0.15, 0.15, 0.15]'))
save(961, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY及SIZE条件030",
     blend_add(sub(sub(b925, A, cond("s_SIZE", "c_condsize") + A), K, K.replace('"]\n', '", "c_condsize"]\n')), "c_condsize", "0.30"))
save(962, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY及BETA条件", add(cond("s_BETA", "c_condbeta"), "c_condbeta", "0.15"))
save(963, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY及MOMENTUM条件", add(cond("s_MOMENTUM", "c_condmom"), "c_condmom", "0.15"))
save(964, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY及RESVOL条件", add(cond("s_RESVOL", "c_condresvol"), "c_condresvol", "0.15"))
save(965, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY及下影线", add(DNSH, "c_dnsh", "0.15"))
save(966, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY及跳空015", add(GAP, "c_gap", "0.15"))
