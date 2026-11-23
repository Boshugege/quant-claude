# -*- coding: utf-8 -*-
"""第一百四十九批：把「IC_IR/压力榜首 816（489+时点重心）」与「多空臂 828（482+时点重心）」对接。
- 841：816 + 隔夜跳空 c_gap 0.30（IC_IR/压力 + 多空）
- 842：816 + c_gap 0.15 + c_utime 0.45
- 843：828 的 _cols 再加隔夜跳空 0.15（同时带市值加权行业滞后与时点重心）
- 844：828 + 2.5 倍探针
- 845：841 + 2.5 倍探针
"""
import pathlib, re, numpy as np
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
UTIME_TAIL = '    df["c_utime"] = -(_cen - 5.0)\n'
GAP_CODE = (
    '    # 隔夜跳空（当日首窗价/昨收，首窗=0925 集合竞价）：反向。\n'
    '    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
    '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
    '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
    '    _mg = _dlyg.set_index(["instrument", "day"])\n'
    '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
    '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:70], t.count(old)); return t.replace(old, new, 1)


def add_keep(s, cols, num):
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S); assert m, num
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body: body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


def emit(num, name, s):
    p = D / ("%s_%s.py" % (num, name)); p.write_text(s, encoding="utf-8"); print("写出", p.name)


S816 = (D / "816_行业市值中性化加时点重心030.py").read_text(encoding="utf-8")
SIG816 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.30])\n'
N20 = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'


def add_gap(s, num):
    s = repl(s, UTIME_TAIL, UTIME_TAIL + GAP_CODE, num + "_gap")
    return add_keep(s, ["c_gap"], num)


# 841 / 842
s = add_gap(S816, "841")
s = repl(s, SIG816, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.30, 0.30])\n', "841_sig")
emit("841", "行业市值中性化时点重心加隔夜跳空", s)

s = add_gap(S816, "842")
s = repl(s, SIG816, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.15])\n', "842_sig")
emit("842", "行业市值中性化时点重心045加隔夜跳空", s)

emit("845", "行业市值中性化时点重心加隔夜跳空二点五", repl(s, N20, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "845"))

# 843 828 + c_gap
S828 = (D / "828_市值加权行业滞后加时点重心二点零.py").read_text(encoding="utf-8")
C828 = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indmwrev", "c_utime"]\n'
W828 = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10, 0.30]), (len(result), 1))\n'
WL828 = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08, 0.20])\n'
s = repl(S828, C828, '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indmwrev", "c_utime", "c_gap"]\n', "843_cols")
s = repl(s, W828, '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10, 0.30, 0.15]), (len(result), 1))\n', "843_w")
s = repl(s, WL828, '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08, 0.20, 0.10])\n', "843_wl")
emit("843", "市值加权行业滞后加时点重心隔夜跳空", s)

emit("844", "市值加权行业滞后加时点重心二点五", repl(S828, N20, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "844"))
