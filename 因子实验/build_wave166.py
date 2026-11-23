# -*- coding: utf-8 -*-
"""第一百六十六批：压低 935/925 支路换手，并把两条新轴接到 IC_IR/压力臂。
- 951：935 + 隔夜跳空 0.10
- 952：935 + 早盘动量反转 0.15
- 953：935 中性化 1.75x
- 954：935 + 首窗成交占比 0.15
- 955：875（833+跳空0.10）+ 换手加权行业内上影线残差 0.15 + LIQUIDTY 条件 0.15
- 956：866（压力榜首）+ LIQUIDTY 条件反转 0.15
- 957：925 把 c_condliq 也纳入中性化特征（诊断）
- 958：902 + 官方 BETA 条件反转 0.15
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
EARLY30 = ('    _pivE = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
           '    _keyE = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
           '    _p10 = _pivE["100000"].reindex(_keyE).to_numpy() if "100000" in _pivE.columns else np.full(len(df), np.nan)\n'
           '    _p09 = _pivE["093000"].reindex(_keyE).to_numpy() if "093000" in _pivE.columns else np.full(len(df), np.nan)\n'
           '    _okE = (_p10 > 0) & (_p09 > 0)\n'
           '    df["c_first30"] = -np.where(_okE, np.log(np.where(_okE, _p10, 1.0) / np.where(_okE, _p09, 1.0)), np.nan)\n')
OPEN30 = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
          '    _daW = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
          '    _o1W = df["sum_amt"].where(_hmsW == "093000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
          '    df["c_open30"] = -roll(_o1W / _daW.where(_daW > 0), 40, "mean", 10)\n')
def cond(col, tag):
    return ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
            '    _qS = pd.qcut(pd.Series(df["%s"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
            '    df["%s"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n' % (col, tag))
UPSHTW = ('    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
          '    _twn = _tw / _tw.groupby([df["date"], df["ind_code"]]).transform("sum").where(lambda x: x > 0)\n'
          '    df["c_upshtwres"] = df["c_upsh"] - (df["c_upsh"] * _twn).groupby([df["date"], df["ind_code"]]).transform("sum")\n')

# 951-954 基于 935
b935 = load(935)
A935 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K935 = 'c_upshtwres", "c_condsize"]\n'
def add935(code, tag, w):
    t = sub(b935, A935, code + A935)
    t = sub(t, K935, K935.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)
save(951, "条件反转扩展中性化加换手加权行业上影残差与SIZE及跳空", add935(GAP, "c_gap", "0.10"))
save(952, "条件反转扩展中性化加换手加权行业上影残差与SIZE及早盘动量", add935(EARLY30, "c_first30", "0.15"))
save(953, "条件反转扩展中性化加换手加权行业上影残差与SIZE一点七五", sub(b935, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 1.75)'))
save(954, "条件反转扩展中性化加换手加权行业上影残差与SIZE及首窗成交", add935(OPEN30, "c_open30", "0.15"))

# 955：875 + c_upshtwres + LIQUIDTY 条件
b875 = load(875)
K875 = '"c_body", "c_utime", "c_gap"]\n'
A833B = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
t955 = sub(b875, A833B, '    df["ind_code"] = _ind.reindex(_sid).to_numpy()\n' + UPSHTW + cond("s_LIQUIDTY", "c_condliq") + A833B)
t955 = sub(t955, K875, '"c_body", "c_utime", "c_gap", "c_upshtwres", "c_condliq"]\n')
t955 = blend_add(blend_add(t955, "c_upshtwres", "0.15"), "c_condliq", "0.15")
save(955, "行业市值中性化时点重心045跳空010加换手加权上影残差与LIQUIDTY条件", t955)

# 956：866 + LIQUIDTY 条件
b866 = load(866)
t956 = sub(b866, A833B, cond("s_LIQUIDTY", "c_condliq") + A833B)
t956 = sub(t956, '"c_body", "c_utime", "c_gap"]\n', '"c_body", "c_utime", "c_gap", "c_condliq"]\n')
save(956, "行业市值中性化时点重心030跳空015加LIQUIDTY条件反转", blend_add(t956, "c_condliq", "0.15"))

# 957：925 把 c_condliq 纳入中性化
b925 = load(925)
save(957, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件诊断",
     sub(b925, 'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"] + sorted([c for c in result.columns if c.startswith("i_")])',
              'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2", "c_condliq"] + sorted([c for c in result.columns if c.startswith("i_")])'))

# 958：902 + BETA 条件
b902 = load(902)
t958 = sub(b902, A935, cond("s_BETA", "c_condbeta") + A935)
t958 = sub(t958, 'c_upshtwres"]\n', 'c_upshtwres", "c_condbeta"]\n')
save(958, "条件反转扩展中性化加换手加权行业上影残差与BETA条件", blend_add(t958, "c_condbeta", "0.15"))
