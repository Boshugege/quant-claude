# -*- coding: utf-8 -*-
"""第一百七十二批：只做新信息源，不改 SQL（零透传风险）。
- 994：成交额时段重心（c_amttime，成交在日内的时间重心）——新源，挂 833 臂
- 995：早盘成交占比（c_mornamt，上午成交额/全天）——新源，挂 833 臂
- 996：成交集中度 HHI（c_amthhi）——新源，挂 IC_IR 榜首 847 臂
- 997：成交集中度 HHI（c_amthhi）——新源，挂压力榜首 895 臂
四个全部是新信息源，无头部小变体。
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

AMTTIME = ('    _tam = (df["date"].dt.hour - 9) + df["date"].dt.minute / 60.0\n'
           '    _snT = (df["sum_amt"] * _tam).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
           '    _sdT = df["sum_amt"].groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
           '    df["c_amttime"] = -((_snT / _sdT.where(_sdT > 0)) - 3.0)\n')
MORNAMT = ('    _hmsM2 = df["date"].dt.strftime("%H%M%S")\n'
           '    _sdM2 = df["sum_amt"].groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
           '    _mvM2 = df["sum_amt"].where((_hmsM2 >= "093000") & (_hmsM2 <= "113000"), 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
           '    df["c_mornamt"] = _mvM2 / _sdM2.where(_sdM2 > 0)\n')
HHI = ('    _pivH = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
       '    _shareH = _pivH.div(_pivH.sum(axis=1).replace(0.0, np.nan), axis=0)\n'
       '    _hhiH = (_shareH.pow(2)).sum(axis=1)\n'
       '    _keyH = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_amthhi"] = -_hhiH.reindex(_keyH).to_numpy()\n')

A833 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K833 = '"c_body", "c_utime"]\n'
save(994, "行业市值中性化时点重心加成交时段重心", add(load(833), AMTTIME, "c_amttime", "0.15", A833, K833))
save(995, "行业市值中性化时点重心加早盘成交占比", add(load(833), MORNAMT, "c_mornamt", "0.15", A833, K833))

t847 = load(847)
save(996, "ICIR榜首847加成交集中度", add(t847, HHI, "c_amthhi", "0.15",
     '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n', '"c_utime", "c_indmwrev"]\n'))
t895 = load(895)
save(997, "压力榜首895加成交集中度", add(t895, HHI, "c_amthhi", "0.15",
     '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n', '"c_utime", "c_gap"]\n'))
