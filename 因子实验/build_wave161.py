# -*- coding: utf-8 -*-
"""第一百六十一批：把「压换手」的两条轻量新轴（首窗成交占比、早盘动量）接到风格条件反转臂与两条榜首臂上。
- 911：870（LIQUIDTY 条件反转）+ 首窗成交占比 0.15
- 912：870 + 早盘动量反转 0.15
- 913：869（SIZE 条件反转）+ 首窗成交占比 0.15
- 914：869 + 早盘动量反转 0.15
- 915：857 臂 + 首窗成交占比 0.30
- 916：857 臂 + 早盘动量反转 0.30
- 917：866（压力榜首）+ 首窗成交占比 0.15
- 918：873（833+早盘动量）权重 0.15 -> 0.30
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

def add_generic(text, anchor, keep_tail, code, tag, w):
    t = sub(text, anchor, code + anchor)
    t = sub(t, keep_tail, keep_tail.replace("]\n", ', "%s"]\n' % tag))
    m = re.search(r'(    result\["signal"\] = rank_blend\(\[)(.*?)(\], \[)(.*?)(\]\))', t)
    assert m, "no rank_blend line"
    return t[:m.start()] + m.group(1) + m.group(2) + ', "%s"' % tag + m.group(3) + m.group(4) + ", %s" % w + m.group(5) + t[m.end():]

# 870 / 869（833 管线：anchor/keep/blend 同 833）
A833 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K870 = '"c_body", "c_utime", "c_condliq"]\n'
B870 = '(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_condliq"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.30])'
K869 = '"c_body", "c_utime", "c_condsize"]\n'
B869 = '(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_condsize"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.30])'
b870, b869 = load(870), load(869)
save(911, "行业市值中性化时点重心加风格LIQUIDTY条件反转与首窗成交", add_generic(b870, A833, K870, OPEN30, "c_open30", "0.15"))
save(912, "行业市值中性化时点重心加风格LIQUIDTY条件反转与早盘动量", add_generic(b870, A833, K870, EARLY30, "c_first30", "0.15"))
save(913, "行业市值中性化时点重心加风格SIZE条件反转与首窗成交", add_generic(b869, A833, K869, OPEN30, "c_open30", "0.15"))
save(914, "行业市值中性化时点重心加风格SIZE条件反转与早盘动量", add_generic(b869, A833, K869, EARLY30, "c_first30", "0.15"))

# 857 臂
A857 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K857 = '"mcap", "ind_code", "c_utime"]\n'
B857 = '(["c_cond_turnself", "c_utime"], [1.0, 0.15])'
b857 = load(857)
save(915, "条件反转扩展中性化加首窗成交占比", add_generic(b857, A857, K857, OPEN30, "c_open30", "0.30"))
save(916, "条件反转扩展中性化加早盘动量", add_generic(b857, A857, K857, EARLY30, "c_first30", "0.30"))

# 866（压力榜首）
A866 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K866 = '"c_body", "c_utime", "c_gap"]\n'
B866 = '(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.15])'
b866 = load(866)
save(917, "行业市值中性化时点重心030跳空015加首窗成交", add_generic(b866, A866, K866, OPEN30, "c_open30", "0.15"))

# 873（833+早盘动量）权重 0.30
b873 = load(873)
save(918, "行业市值中性化时点重心加早盘动量反转030", sub(b873, '"c_utime", "c_first30"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.15]',
                                                        '"c_utime", "c_first30"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.30]'))
