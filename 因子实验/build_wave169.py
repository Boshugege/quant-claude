# -*- coding: utf-8 -*-
"""第一百六十九批：多空榜首 960 与 IC 均值榜首 953 的收尾叠加。
- 975：960 + 隔夜跳空 0.10
- 976：960 + 早盘动量反转 0.15
- 977：960 + 首窗成交占比 0.15
- 978：960 中性化 1.75x
- 979：960 + 官方 SIZE 条件反转 0.15
- 980：953 + 官方 RESVOL 条件反转 0.15
- 981：895（816+跳空0.10）+ 官方 LIQUIDTY 条件反转 0.15
- 982：875（833+跳空0.10）+ 官方 RESVOL 条件反转 0.15
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

b960 = load(960)
A = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K = 'c_upshtwres", "c_condliq"]\n'
def add(code, tag, w):
    t = sub(b960, A, code + A)
    t = sub(t, K, K.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)
save(975, "多空榜首960加隔夜跳空010", add(GAP, "c_gap", "0.10"))
save(976, "多空榜首960加早盘动量", add(EARLY30, "c_first30", "0.15"))
save(977, "多空榜首960加首窗成交", add(OPEN30, "c_open30", "0.15"))
save(978, "多空榜首960一点七五", sub(b960, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 1.75)'))
save(979, "多空榜首960加SIZE条件", add(cond("s_SIZE", "c_condsize"), "c_condsize", "0.15"))

b953 = load(953)
A953 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K953 = 'c_upshtwres", "c_condsize"]\n'
def add953(code, tag, w):
    t = sub(b953, A953, code + A953)
    t = sub(t, K953, K953.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)
save(980, "IC均值榜首953加RESVOL条件", add953(cond("s_RESVOL", "c_condresvol"), "c_condresvol", "0.15"))

# 981/982：833 管线（需先补 ind_code 再算条件轴）——用 c_avgtr 锚点
A833B = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
for num, base, name, tag in [(981, 895, "IC_IR臂895加LIQUIDTY条件", "c_condliq"), (982, 875, "IC_IR臂875加RESVOL条件", "c_condresvol")]:
    b = load(base)
    t = sub(b, A833B, cond("s_LIQUIDTY" if tag == "c_condliq" else "s_RESVOL", tag) + A833B)
    t = sub(t, '"c_body", "c_utime", "c_gap"]\n', '"c_body", "c_utime", "c_gap", "%s"]\n' % tag)
    save(num, name, blend_add(t, tag, "0.15"))
