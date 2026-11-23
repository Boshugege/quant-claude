# -*- coding: utf-8 -*-
"""第一百五十批：在 833（IC_IR 榜首 1.495）管线上并多空分量。
- 846：833 + 隔夜跳空 0.30
- 847：833 + 市值加权行业滞后 0.10（给 816 管线补 ind_code）
- 848：833 + 2.25 倍探针
- 849：828 + 隔夜跳空 0.30
- 850：833 + 隔夜跳空 0.15 + 市值加权行业滞后 0.10
"""
import pathlib, re
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
UTIME_TAIL = '    df["c_utime"] = -(_cen - 5.0)\n'
GAP_CODE = (
    '    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
    '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
    '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
    '    _mg = _dlyg.set_index(["instrument", "day"])\n'
    '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
    '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
INDM_CODE = (
    '    df["ind_code"] = _ind.reindex(_sid).astype(str).to_numpy()\n'
    '    _dlyb = df.groupby(["day", "instrument"], sort=False).agg(\n'
    '        _last=("last_px", "last"), _pc=("pre_close", "first"), _ind=("ind_code", "first"), _mc=("mcap", "first")).reset_index()\n'
    '    _dlyb["_r"] = np.where((_dlyb["_last"] > 0) & (_dlyb["_pc"] > 0), np.log(_dlyb["_last"] / _dlyb["_pc"]), np.nan)\n'
    '    _ws = _dlyb.groupby(["day", "_ind"], sort=False)["_mc"].transform("sum")\n'
    '    _dlyb["_wr"] = _dlyb["_r"] * (_dlyb["_mc"] / _ws.where(_ws > 0))\n'
    '    _idp = _dlyb.groupby(["day", "_ind"], sort=False)["_wr"].sum().unstack("_ind").sort_index().shift(1)\n'
    '    _flat = _idp.stack(dropna=False)\n'
    '    _ikey = pd.MultiIndex.from_arrays([df["day"].values, df["ind_code"].values])\n'
    '    df["c_indmwrev"] = -_flat.reindex(_ikey).to_numpy()\n')
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

S833 = (D / "833_行业市值中性化加时点重心045.py").read_text(encoding="utf-8")
SIG833 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45])\n'
N20 = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'

s = repl(S833, UTIME_TAIL, UTIME_TAIL + GAP_CODE, "846_gap"); s = add_keep(s, ["c_gap"], "846")
s = repl(s, SIG833, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.30])\n', "846_sig")
emit("846", "行业市值中性化时点重心045加隔夜跳空030", s)

s = repl(S833, UTIME_TAIL, UTIME_TAIL + INDM_CODE, "847_indm"); s = add_keep(s, ["c_indmwrev"], "847")
s = repl(s, SIG833, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_indmwrev"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.10])\n', "847_sig")
emit("847", "行业市值中性化时点重心045加行业市值滞后", s)

emit("848", "行业市值中性化时点重心045二点二五", repl(S833, N20, '    result["factor"] = neutralize("signal", CHARS4, 2.25)\n', "848"))

s = repl(S833, UTIME_TAIL, UTIME_TAIL + GAP_CODE + INDM_CODE, "850_both"); s = add_keep(s, ["c_gap", "c_indmwrev"], "850")
s = repl(s, SIG833, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap", "c_indmwrev"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.15, 0.10])\n', "850_sig")
emit("850", "行业市值中性化时点重心045加跳空与行业市值滞后", s)

S828 = (D / "828_市值加权行业滞后加时点重心二点零.py").read_text(encoding="utf-8")
C828 = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indmwrev", "c_utime"]\n'
W828 = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10, 0.30]), (len(result), 1))\n'
WL828 = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08, 0.20])\n'
s = repl(S828, C828, '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indmwrev", "c_utime", "c_gap"]\n', "849_cols")
s = repl(s, W828, '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10, 0.30, 0.30]), (len(result), 1))\n', "849_w")
s = repl(s, WL828, '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08, 0.20, 0.20])\n', "849_wl")
emit("849", "市值加权行业滞后加时点重心隔夜跳空030", s)
