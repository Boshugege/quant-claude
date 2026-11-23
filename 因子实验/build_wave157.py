# -*- coding: utf-8 -*-
"""第一百五十七批：在 856/857（条件反转×扩展中性化，多空 47.30 / IC 均值 0.0799）主臂上继续开轴。
- 879：+隔夜跳空 0.15
- 880：时点重心 0.15 -> 0.45
- 881：+尾盘 30 分钟动量反转 0.15
- 882：+官方 SIZE 五分位条件反转 0.30
- 883：+官方 LIQUIDTY 五分位条件反转 0.30
- 884：+首窗成交占比 0.15
- 885：+隔夜跳空 0.15 且时点重心 0.45（组合）
- 886：+下影线占比 0.15
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
    assert old in t, old[:100]
    return t.replace(old, new, 1)

b857 = load(857)
ANCHOR = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
KEEP_TAIL = '"mcap", "ind_code", "c_utime"]\n'

GAP = ('    # 隔夜跳空（当日首窗价/昨收，首窗=0925 集合竞价）：反向。\n'
       '    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
LATE = ('    _pivW = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
        '    _keyW = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
        '    _p15 = _pivW["150000"].reindex(_keyW).to_numpy() if "150000" in _pivW.columns else np.full(len(df), np.nan)\n'
        '    _p14 = _pivW["143000"].reindex(_keyW).to_numpy() if "143000" in _pivW.columns else np.full(len(df), np.nan)\n'
        '    _okW = (_p15 > 0) & (_p14 > 0)\n'
        '    df["c_last30"] = -np.where(_okW, np.log(np.where(_okW, _p15, 1.0) / np.where(_okW, _p14, 1.0)), np.nan)\n')
OPEN30 = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
          '    _daW = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
          '    _o1W = df["sum_amt"].where(_hmsW == "093000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
          '    df["c_open30"] = -roll(_o1W / _daW.where(_daW > 0), 40, "mean", 10)\n')

def cond_block(col, tag):
    return ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
            '    _qS = pd.qcut(pd.Series(df["%s"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
            '    df["%s"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n' % (col, tag))

def add_axis(text, code, tag, weight, extra=None):
    t = sub(text, ANCHOR, code + ANCHOR)
    t = sub(t, KEEP_TAIL, '"mcap", "ind_code", "c_utime", "%s"]\n' % tag)
    olds = '["c_cond_turnself", "c_utime"], [1.0, 0.15]'
    news = '["c_cond_turnself", "c_utime", "%s"], [1.0, 0.15, %s]' % (tag, weight)
    if extra:
        news = '["c_cond_turnself", "c_utime", "%s", "%s"], [1.0, 0.15, %s, %s]' % (tag, extra[0], weight, extra[1])
    t = sub(t, olds, news)
    return t

# 879 gap / 885 gap+utime0.45
save(879, "条件反转扩展中性化加跳空", add_axis(b857, GAP, "c_gap", "0.15"))
t885 = add_axis(b857, GAP, "c_gap", "0.15")
t885 = sub(t885, '["c_cond_turnself", "c_utime", "c_gap"], [1.0, 0.15, 0.15]', '["c_cond_turnself", "c_utime", "c_gap"], [1.0, 0.45, 0.15]')
save(885, "条件反转扩展中性化加跳空与时点重心045", t885)

# 880 utime 0.45
save(880, "条件反转扩展中性化加时点重心045", sub(b857, '["c_cond_turnself", "c_utime"], [1.0, 0.15]', '["c_cond_turnself", "c_utime"], [1.0, 0.45]'))

save(881, "条件反转扩展中性化加尾盘动量", add_axis(b857, LATE, "c_last30", "0.15"))
save(882, "条件反转扩展中性化加风格SIZE条件反转", add_axis(b857, cond_block("s_SIZE", "c_condsize"), "c_condsize", "0.30"))
save(883, "条件反转扩展中性化加风格LIQUIDTY条件反转", add_axis(b857, cond_block("s_LIQUIDTY", "c_condliq"), "c_condliq", "0.30"))
save(884, "条件反转扩展中性化加首窗成交占比", add_axis(b857, OPEN30, "c_open30", "0.15"))

# 886 下影线占比（857 管线已含 sum_dnsh，无需改 SQL）
save(886, "条件反转扩展中性化加下影线占比", add_axis(b857, '    _dnsh = df["sum_dnsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_dnsh"] = -roll(_dnsh, 40, "mean", 10)\n', "c_dnsh", "0.15"))
