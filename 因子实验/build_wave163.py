# -*- coding: utf-8 -*-
"""第一百六十三批：已验证杠杆的组合 + 两个日内分布新轴。
- 927：866（压力榜首）+ 换手加权行业内上影线残差 0.15
- 928：902 + 隔夜跳空 0.10
- 929：875（833+跳空0.10）+ 官方 LIQUIDTY 条件反转 0.15
- 930：875 + 换手加权行业内上影线残差 0.15
- 931：833 + 下午/上午成交额比（新轴）
- 932：833 + 日内成交量分布集中度（1-熵，新轴）
- 933：902 仅 4 特征中性化（诊断）
- 934：902 中性化 1.75x
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
GAP = ('    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
# 下午/上午成交额比：早盘窗口（093000-113000 段）与午后窗口（133000-150000 段）
PMAM = ('    _hmsP = df["date"].dt.strftime("%H%M%S")\n'
        '    _amP = df["sum_amt"].where(_hmsP <= "113000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
        '    _pmP = df["sum_amt"].where(_hmsP >= "130000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
        '    df["c_pmam"] = -roll(_pmP / _amP.where(_amP > 0), 40, "mean", 10)\n')
# 日内成交量分布集中度 = 1 - 归一化熵（高=集中在少数窗口）
VENT = ('    _hmsV = df["date"].dt.strftime("%H%M%S")\n'
        '    _dV = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
        '    _pV = (df["sum_amt"].where(_hmsV <= "150000", 0.0) / _dV.where(_dV > 0)).clip(lower=0.0)\n'
        '    _entV = -(_pV * np.log(_pV.where(_pV > 0))).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
        '    df["c_vent"] = -roll(_entV, 40, "mean", 10)\n')
CONDLIQ = ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
           '    _qS = pd.qcut(pd.Series(df["s_LIQUIDTY"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
           '    df["c_condliq"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n')

def add_axis(text, anchor, keep_tail, code, tag, w):
    t = sub(text, anchor, code + anchor)
    t = sub(t, keep_tail, keep_tail.replace("]\n", ', "%s"]\n' % tag))
    m = re.search(r'(    result\["signal"\] = rank_blend\(\[)(.*?)(\], \[)(.*?)(\]\))', t)
    assert m
    return t[:m.start()] + m.group(1) + m.group(2) + ', "%s"' % tag + m.group(3) + m.group(4) + ", %s" % w + m.group(5) + t[m.end():]

# 927：866 + c_upshtwres（866 无 ind_code 列，需先补）
b866 = load(866)
A866B = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K866 = '"c_body", "c_utime", "c_gap"]\n'
t927 = sub(b866, A866B, '    df["ind_code"] = _ind.reindex(_sid).to_numpy()\n' + UPSHTW + A866B)
save(927, "行业市值中性化时点重心030跳空015加换手加权行业上影残差", add_axis(t927, A866B, K866, '', "c_upshtwres", "0.15"))

# 928：902 + 跳空 0.10
b902 = load(902)
A902 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K902 = '"mcap", "ind_code", "c_utime", "c_upshtwres"]\n'
save(928, "条件反转扩展中性化加换手加权行业上影残差与跳空010", add_axis(b902, A902, K902, GAP, "c_gap", "0.10"))

# 929/930：875（833+跳空0.10）
b875 = load(875)
A833 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K875 = '"c_body", "c_utime", "c_gap"]\n'
save(929, "行业市值中性化时点重心045跳空010加风格LIQUIDTY条件反转", add_axis(b875, A833, K875, CONDLIQ, "c_condliq", "0.15"))
A833B = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
t930 = sub(b875, A833B, '    df["ind_code"] = _ind.reindex(_sid).to_numpy()\n' + UPSHTW + A833B)
save(930, "行业市值中性化时点重心045跳空010加换手加权行业上影残差", add_axis(t930, A833B, K875, '', "c_upshtwres", "0.15"))

# 931/932：833 日内分布新轴
b833 = load(833)
K833 = '"c_body", "c_utime"]\n'
save(931, "行业市值中性化时点重心加下午上午成交比", add_axis(b833, A833, K833, PMAM, "c_pmam", "0.15"))
save(932, "行业市值中性化时点重心加日内成交集中度", add_axis(b833, A833, K833, VENT, "c_vent", "0.15"))

# 933/934：902 诊断与细网格
save(933, "条件反转扩展中性化加换手加权行业上影残差四特征诊断", sub(b902, 'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"] + sorted([c for c in result.columns if c.startswith("i_")])', 'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]'))
save(934, "条件反转扩展中性化加换手加权行业上影残差一点七五", sub(b902, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 1.75)'))
