# -*- coding: utf-8 -*-
"""第一百五十九批：压力维度（隔夜跳空权重网格）与两条新源轴（换手排位变化、换手加权行业内上影线残差）。
- 895：816 管线（c_utime 0.30）+ 隔夜跳空 0.10
- 896：816 管线 + 隔夜跳空 0.20
- 897：866（跳空 0.15）2.25 倍中性化
- 898：860（条件反转 + c_utime 0.30）+ 隔夜跳空 0.15
- 899：833 + 换手率排位变化 0.10（新源）
- 900：833 + 换手加权行业内上影线残差 0.15（新源）
- 901：857 + 换手率排位变化 0.10（新源，多空/IC 均值臂）
- 902：857 + 换手加权行业内上影线残差 0.15（新源，多空/IC 均值臂）
"""
import io, os

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

# ---- 895/896/897：压力网格，基 866（时点重心0.30+跳空0.15） ----
b866 = load(866)
G866 = '["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15]'
save(895, "行业市值中性化时点重心030跳空010", sub(b866, G866, G866.replace("0.30, 0.15]", "0.30, 0.10]")))
save(896, "行业市值中性化时点重心030跳空020", sub(b866, G866, G866.replace("0.30, 0.15]", "0.30, 0.20]")))
save(897, "行业市值中性化时点重心030跳空015二点二五", sub(b866, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 2.25)'))

# ---- 898：条件反转臂 + 跳空 + 时点重心 0.30 ----
b860 = load(860)
GAP = ('    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
t = sub(b860, '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n', GAP + '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n')
t = sub(t, '"mcap", "ind_code", "c_utime"]\n', '"mcap", "ind_code", "c_utime", "c_gap"]\n')
t = sub(t, '["c_cond_turnself", "c_utime"], [1.0, 0.30]', '["c_cond_turnself", "c_utime", "c_gap"], [1.0, 0.30, 0.15]')
save(898, "条件反转扩展中性化加跳空与时点重心030", t)

# ---- 899/901：换手率排位变化 ----
TURNRK = ('    _dlyT = df.groupby(["day", "instrument"], sort=False).agg(\n'
          '        _amt=("sum_amt", "sum"), _mc=("mcap", "first")).reset_index()\n'
          '    _dlyT["_turn"] = _dlyT["_amt"] / _dlyT["_mc"].where(_dlyT["_mc"] > 0)\n'
          '    _dlyT = _dlyT.sort_values(["instrument", "day"]).reset_index(drop=True)\n'
          '    _dlyT["_trk"] = _dlyT.groupby("day", sort=False)["_turn"].rank(pct=True)\n'
          '    _dlyT["_chg"] = _dlyT["_trk"] - _dlyT.groupby("instrument", sort=False)["_trk"].shift(1)\n'
          '    _mT = _dlyT.set_index(["instrument", "day"])\n'
          '    _kT = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
          '    df["c_turnrkchg"] = -_mT["_chg"].reindex(_kT).to_numpy()\n')
# ---- 900/902：换手加权行业内上影线残差 ----
UPSHTW = ('    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
          '    _twn = _tw / _tw.groupby([df["date"], df["ind_code"]]).transform("sum").where(lambda x: x > 0)\n'
          '    df["c_upshtwres"] = df["c_upsh"] - (df["c_upsh"] * _twn).groupby([df["date"], df["ind_code"]]).transform("sum")\n')

b833 = load(833)
A833 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K833 = '"c_body", "c_utime"]\n'
B833 = '["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45]'

def add833(text, code, tag, w):
    t = sub(text, A833, code + A833)
    t = sub(t, K833, '"c_body", "c_utime", "%s"]\n' % tag)
    t = sub(t, B833, B833.replace('"c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45]',
                                   '"c_utime", "%s"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, %s]' % (tag, w)))
    return t

save(899, "行业市值中性化时点重心加换手排位变化", add833(b833, TURNRK, "c_turnrkchg", "0.10"))
UPSHTW833 = '    df["ind_code"] = _ind.reindex(_sid).to_numpy()\n' + UPSHTW
A833B = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
t900 = sub(b833, A833B, UPSHTW833 + A833B)
t900 = sub(t900, K833, '"c_body", "c_utime", "c_upshtwres"]\n')
t900 = sub(t900, B833, B833.replace('"c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45]',
                                    '"c_utime", "c_upshtwres"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.15]'))
save(900, "行业市值中性化时点重心加换手加权行业上影残差", t900)

b857 = load(857)
A857 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K857 = '"mcap", "ind_code", "c_utime"]\n'
B857 = '["c_cond_turnself", "c_utime"], [1.0, 0.15]'

def add857(text, code, tag, w):
    t = sub(text, A857, code + A857)
    t = sub(t, K857, '"mcap", "ind_code", "c_utime", "%s"]\n' % tag)
    t = sub(t, B857, '["c_cond_turnself", "c_utime", "%s"], [1.0, 0.15, %s]' % (tag, w))
    return t

save(901, "条件反转扩展中性化加换手排位变化", add857(b857, TURNRK, "c_turnrkchg", "0.10"))
save(902, "条件反转扩展中性化加换手加权行业上影残差", add857(b857, UPSHTW, "c_upshtwres", "0.15"))
