# -*- coding: utf-8 -*-
"""第一百六十四批：围绕 902（三线齐升）叠加其余已验证新轴。
- 935：902 + 官方 SIZE 条件反转 0.15
- 936：902 + 行业内条件反转 0.15
- 937：902 + 行业内市值加权同伴残差 0.10
- 938：902 + 时点重心 0.45
- 939：902 + 隔夜跳空 0.30
- 940：833 臂 + 换手加权行业内上影线残差 0.15（IC_IR 臂上找甜点）
- 941：866（压力榜首）+ 换手加权行业内上影线残差 0.30
- 942：902 + 首窗成交占比 0.30
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

def add_axis(text, anchor, keep_tail, code, tag, w):
    t = sub(text, anchor, code + anchor)
    t = sub(t, keep_tail, keep_tail.replace("]\n", ', "%s"]\n' % tag))
    return blend_add(t, tag, w)

UPSHTW = ('    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
          '    _twn = _tw / _tw.groupby([df["date"], df["ind_code"]]).transform("sum").where(lambda x: x > 0)\n'
          '    df["c_upshtwres"] = df["c_upsh"] - (df["c_upsh"] * _twn).groupby([df["date"], df["ind_code"]]).transform("sum")\n')
OPEN30 = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
          '    _daW = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
          '    _o1W = df["sum_amt"].where(_hmsW == "093000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
          '    df["c_open30"] = -roll(_o1W / _daW.where(_daW > 0), 40, "mean", 10)\n')
GAP = ('    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
CONDLIQ = ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
           '    _qS = pd.qcut(pd.Series(df["s_LIQUIDTY"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
           '    df["c_condliq"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n')
CONDSIZE = CONDLIQ.replace("s_LIQUIDTY", "s_SIZE").replace("c_condliq", "c_condsize")

b902 = load(902)
A902 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K902 = '"mcap", "ind_code", "c_utime", "c_upshtwres"]\n'
save(935, "条件反转扩展中性化加换手加权行业上影残差与SIZE条件", add_axis(b902, A902, K902, CONDSIZE, "c_condsize", "0.15"))
save(936, "条件反转扩展中性化加换手加权行业上影残差与行业内条件", blend_add(sub(b902, A902, A902), "c_condrev", "0.15"))
save(937, "条件反转扩展中性化加换手加权行业上影残差与行业内市值加权同伴", add_axis(b902, A902, K902,
     '    _dly_b = df.groupby(["day", "instrument"], sort=False).agg(\n'
     '        _last=("last_px", "last"), _pc=("pre_close", "first"), _ind=("ind_code", "first"), _mc=("mcap", "first")).reset_index()\n'
     '    _dly_b["_r"] = np.where((_dly_b["_last"] > 0) & (_dly_b["_pc"] > 0), np.log(_dly_b["_last"] / _dly_b["_pc"]), np.nan)\n'
     '    _wsum = _dly_b.groupby(["day", "_ind"], sort=False)["_mc"].transform("sum")\n'
     '    _dly_b["_wr"] = _dly_b["_r"] * (_dly_b["_mc"] / _wsum.where(_wsum > 0))\n'
     '    _idp = _dly_b.groupby(["day", "_ind"], sort=False)["_wr"].sum().unstack("_ind").sort_index().shift(1)\n'
     '    _flat = _idp.stack(dropna=False)\n'
     '    _ikey = pd.MultiIndex.from_arrays([df["day"].values, df["ind_code"].values])\n'
     '    df["c_indmwrev"] = -_flat.reindex(_ikey).to_numpy()\n', "c_indmwrev", "0.10"))
save(938, "条件反转扩展中性化加换手加权行业上影残差与时点重心045", sub(b902, '["c_cond_turnself", "c_utime", "c_upshtwres"], [1.0, 0.15, 0.15]',
                                                                        '["c_cond_turnself", "c_utime", "c_upshtwres"], [1.0, 0.45, 0.15]'))
save(939, "条件反转扩展中性化加换手加权行业上影残差与跳空030", sub(add_axis(b902, A902, K902, GAP, "c_gap", "0.30"), '0.15, 0.15, 0.30]', '0.15, 0.15, 0.30]'))
save(942, "条件反转扩展中性化加换手加权行业上影残差与首窗成交030", sub(add_axis(b902, A902, K902, OPEN30, "c_open30", "0.15"), '"c_open30"], [1.0, 0.15, 0.15, 0.15]', '"c_open30"], [1.0, 0.15, 0.15, 0.30]'))

# 940：833 + c_upshtwres 0.15
b833 = load(833)
A833B = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K833 = '"c_body", "c_utime"]\n'
t940 = sub(b833, A833B, '    df["ind_code"] = _ind.reindex(_sid).to_numpy()\n' + UPSHTW + A833B)
save(940, "行业市值中性化时点重心加换手加权行业上影残差015", add_axis(t940, UPSHTW, K833, '', "c_upshtwres", "0.15"))

# 941：866 + c_upshtwres 0.30
b866 = load(866)
t941 = sub(b866, A833B, '    df["ind_code"] = _ind.reindex(_sid).to_numpy()\n' + UPSHTW + A833B)
save(941, "行业市值中性化时点重心030跳空015加换手加权行业上影残差030", add_axis(t941, UPSHTW, '"c_body", "c_utime", "c_gap"]\n', '', "c_upshtwres", "0.30"))
