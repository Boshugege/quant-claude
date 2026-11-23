# -*- coding: utf-8 -*-
"""第一百六十二批：围绕 902（换手加权行业内上影线残差 = 三线齐升的新轴）做权重网格与叠加。
- 919：857 臂 + 换手加权行业内上影线残差 0.10
- 920：857 臂 + 换手加权行业内上影线残差 0.30
- 921：833 臂 + 换手加权行业内上影线残差 0.30（IC_IR 臂）
- 922：902 + 隔夜跳空 0.15
- 923：902 + 中性化 2.25x
- 924：902 + 时点重心 0.30
- 925：902 + 官方 LIQUIDTY 条件反转 0.30（两条最强新轴叠加）
- 926：902 + 首窗成交占比 0.15（压换手）
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

def cond_block(col, tag):
    return ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
            '    _qS = pd.qcut(pd.Series(df["%s"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
            '    df["%s"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n' % (col, tag))

def add_axis(text, anchor, keep_tail, code, tag, w):
    t = sub(text, anchor, code + anchor)
    t = sub(t, keep_tail, keep_tail.replace("]\n", ', "%s"]\n' % tag))
    m = re.search(r'(    result\["signal"\] = rank_blend\(\[)(.*?)(\], \[)(.*?)(\]\))', t)
    assert m
    return t[:m.start()] + m.group(1) + m.group(2) + ', "%s"' % tag + m.group(3) + m.group(4) + ", %s" % w + m.group(5) + t[m.end():]

A857 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K857 = '"mcap", "ind_code", "c_utime"]\n'
b857 = load(857)
save(919, "条件反转扩展中性化加换手加权行业上影残差010", add_axis(b857, A857, K857, UPSHTW, "c_upshtwres", "0.10"))
save(920, "条件反转扩展中性化加换手加权行业上影残差030", add_axis(b857, A857, K857, UPSHTW, "c_upshtwres", "0.30"))

# 921：833 臂（c_upsh 定义在 c_avgtr 之前，锚点要用 c_avgtr 行）
b833 = load(833)
A833B = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K833 = '"c_body", "c_utime"]\n'
t921 = sub(b833, A833B, '    df["ind_code"] = _ind.reindex(_sid).to_numpy()\n' + UPSHTW + A833B)
save(921, "行业市值中性化时点重心加换手加权行业上影残差030", add_axis(t921, A833B, K833, '', "c_upshtwres", "0.30"))
# 修正：add_axis 会再插一次空代码（无影响），但 c_upshtwres 需在 keep 中（已由 add_axis 完成）

# 922-926：基 902
b902 = load(902)
K902 = '"mcap", "ind_code", "c_utime", "c_upshtwres"]\n'
A902 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
save(922, "条件反转扩展中性化加换手加权行业上影残差与跳空", add_axis(b902, A902, K902, GAP, "c_gap", "0.15"))
save(923, "条件反转扩展中性化加换手加权行业上影残差二点二五", sub(b902, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 2.25)'))
save(924, "条件反转扩展中性化加换手加权行业上影残差与时点重心030", sub(b902, '["c_cond_turnself", "c_utime", "c_upshtwres"], [1.0, 0.15, 0.15]', '["c_cond_turnself", "c_utime", "c_upshtwres"], [1.0, 0.30, 0.15]'))
save(925, "条件反转扩展中性化加换手加权行业上影残差与LIQUIDTY条件", add_axis(b902, A902, K902, cond_block("s_LIQUIDTY", "c_condliq"), "c_condliq", "0.30"))
save(926, "条件反转扩展中性化加换手加权行业上影残差与首窗成交", add_axis(b902, A902, K902, OPEN30, "c_open30", "0.15"))
