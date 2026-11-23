# -*- coding: utf-8 -*-
"""第一百七十一批：补队列到15并继续新源探索。
- 991/993：日内逐窗成交额集中度（HHI，突发性 vs 均匀性）——新信息源，挂在 833 臂与 925 臂
- 992：983 下影线时点重心权重 0.15→0.30 的头部小变体（本批唯一头部变体）
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

# 日内逐窗成交额集中度：份额平方和（HHI）。取负，使「均匀分散」得高分。
HHI = ('    _pivH = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="sum_amt", aggfunc="sum")\n'
       '    _shareH = _pivH.div(_pivH.sum(axis=1).replace(0.0, np.nan), axis=0)\n'
       '    _hhiH = (_shareH.pow(2)).sum(axis=1)\n'
       '    _keyH = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_amthhi"] = -_hhiH.reindex(_keyH).to_numpy()\n')

# ---------- 833 臂 ----------
b833 = load(833)
A833 = '    df["c_avgtr"] = roll(_avgtr, 40, "mean", 10)\n'
K833 = '"c_body", "c_utime"]\n'

def add833(text, code, tag, w, anchor=A833):
    t = sub(text, anchor, code + anchor)
    t = sub(t, K833, '"c_body", "c_utime", "%s"]\n' % tag)
    return blend_add(t, tag, w)

save(991, "行业市值中性化时点重心加成交集中度", add833(b833, HHI, "c_amthhi", "0.15"))

# ---------- 992：983 的下影线时点重心权重 0.15→0.30 ----------
t992 = load(983)
t992 = sub(t992, ", 0.45, 0.15])", ", 0.45, 0.30])")
save(992, "行业市值中性化时点重心加下影线时点重心030", t992)

# ---------- 925 臂 ----------
b925 = load(925)
A925 = '    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)\n'
K925 = 'c_upshtwres", "c_condliq"]\n'

def add925(code, tag, w, base=None):
    t = sub(base if base is not None else b925, A925, code + A925)
    t = sub(t, K925, K925.replace('"]\n', '", "%s"]\n' % tag))
    return blend_add(t, tag, w)

save(993, "多空臂925加成交集中度", add925(HHI, "c_amthhi", "0.15"))
