# -*- coding: utf-8 -*-
"""第一百七十七批：行业层面隔夜跳空结构（新信息族，纯 pandas，无 SQL 改动）。
- 1014：925 臂 + 行业内隔夜跳空离散度 c_indgapdisp
- 1015：925 臂 + 行业内隔夜跳空均值 c_indgapmean
- 1016：847 臂 + 行业内隔夜跳空均值 c_indgapmean
- 1017：847 臂 + 行业内隔夜跳空离散度 c_indgapdisp
全部新信息源（行业层面，而非个股日内形状）。
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

GAP_BASE = ('    _hmsg = df["date"].dt.strftime("%H%M%S")\n'
            '    _dop = df["last_px"].where(_hmsg == "093000").groupby([df["instrument"], df["day"]], sort=False).transform("max")\n'
            '    _gpr = np.where((_dop > 0) & (df["pre_close"] > 0), np.log(np.where(_dop > 0, _dop, 1.0) / np.where(df["pre_close"] > 0, df["pre_close"], 1.0)), np.nan)\n'
            '    _gps = pd.Series(_gpr, index=df.index)\n')
INDGAPDISP = GAP_BASE + '    df["c_indgapdisp"] = _gps.groupby([df["date"], df["ind_code"]]).transform("std")\n'
INDGAPMEAN = GAP_BASE + '    df["c_indgapmean"] = _gps.groupby([df["date"], df["ind_code"]]).transform("mean")\n'

A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
A847 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'

def add925(code, tag, w):
    t = sub(load(925), A925, code + A925)
    t = sub(t, 'c_upshtwres", "c_condliq"]\n', 'c_upshtwres", "c_condliq", "%s"]\n' % tag)
    return blend_add(t, tag, w)

def add847(code, tag, w):
    t = sub(load(847), A847, code + A847)
    t = sub(t, '"c_utime", "c_indmwrev"]\n', '"c_utime", "c_indmwrev", "%s"]\n' % tag)
    return blend_add(t, tag, w)

save(1014, "多空臂925加行业隔夜跳空离散度", add925(INDGAPDISP, "c_indgapdisp", "0.15"))
save(1015, "多空臂925加行业隔夜跳空均值", add925(INDGAPMEAN, "c_indgapmean", "0.15"))
save(1016, "ICIR臂847加行业隔夜跳空均值", add847(INDGAPMEAN, "c_indgapmean", "0.15"))
save(1017, "ICIR臂847加行业隔夜跳空离散度", add847(INDGAPDISP, "c_indgapdisp", "0.15"))
