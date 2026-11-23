# -*- coding: utf-8 -*-
"""第一百六十批：风格条件反转轴的扩张与组合（869/870 实测 IC 均值 0.0793/0.0794、多空 45.3/45.4）。
- 903：857 臂 + 官方 LIQUIDTY 条件反转 0.30
- 904：857 臂 + 官方 SIZE 条件反转 0.30
- 905：833 臂 + 官方 BETA 条件反转 0.30
- 906：833 臂 + 官方 MOMENTUM 条件反转 0.30
- 907：833 臂 + 官方 BTOP 条件反转 0.30
- 908：833 臂 + LIQUIDTY 条件反转 0.15（权重下探）
- 909：833 臂 + LIQUIDTY 条件反转 0.60（权重上探）
- 910：870（LIQUIDTY 条件反转 0.30）+ 隔夜跳空 0.15（压力维度）
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

def cond_block(col, tag):
    return ('    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5\n'
            '    _qS = pd.qcut(pd.Series(df["%s"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")\n'
            '    df["%s"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")\n' % (col, tag))

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

save(905, "行业市值中性化时点重心加风格BETA条件反转", add833(b833, cond_block("s_BETA", "c_condbeta"), "c_condbeta", "0.30"))
save(906, "行业市值中性化时点重心加风格MOMENTUM条件反转", add833(b833, cond_block("s_MOMENTUM", "c_condmom"), "c_condmom", "0.30"))
save(907, "行业市值中性化时点重心加风格BTOP条件反转", add833(b833, cond_block("s_BTOP", "c_condbtop"), "c_condbtop", "0.30"))
save(908, "行业市值中性化时点重心加风格LIQUIDTY条件反转015", add833(b833, cond_block("s_LIQUIDTY", "c_condliq"), "c_condliq", "0.15"))
save(909, "行业市值中性化时点重心加风格LIQUIDTY条件反转060", add833(b833, cond_block("s_LIQUIDTY", "c_condliq"), "c_condliq", "0.60"))

b857 = load(857)
A857 = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
K857 = '"mcap", "ind_code", "c_utime"]\n'
B857 = '["c_cond_turnself", "c_utime"], [1.0, 0.15]'

def add857(text, code, tag, w):
    t = sub(text, A857, code + A857)
    t = sub(t, K857, '"mcap", "ind_code", "c_utime", "%s"]\n' % tag)
    t = sub(t, B857, '["c_cond_turnself", "c_utime", "%s"], [1.0, 0.15, %s]' % (tag, w))
    return t

save(903, "条件反转扩展中性化加风格LIQUIDTY条件反转", add857(b857, cond_block("s_LIQUIDTY", "c_condliq"), "c_condliq", "0.30"))
save(904, "条件反转扩展中性化加风格SIZE条件反转", add857(b857, cond_block("s_SIZE", "c_condsize"), "c_condsize", "0.30"))

# 910：870 + 隔夜跳空 0.15
b870 = load(870)
GAP = ('    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
t = sub(b870, A833, GAP + A833)
t = sub(t, '"c_body", "c_utime", "c_condliq"]\n', '"c_body", "c_utime", "c_condliq", "c_gap"]\n')
t = sub(t, '"c_utime", "c_condliq"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.30]',
           '"c_utime", "c_condliq", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.30, 0.15]')
save(910, "行业市值中性化时点重心加风格LIQUIDTY条件反转与跳空", t)
