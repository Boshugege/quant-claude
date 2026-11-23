# -*- coding: utf-8 -*-
"""第一百四十八批：多空臂（823/824）叠时点重心，及 816 的换手/平滑邻点。
- 838：823（头部+隔夜跳空 2.0×）+ c_utime 0.45
- 839：824（头部+市值加权行业滞后 2.0×）+ c_utime 0.30
- 840：816 信号 span10 平滑（换手臂）
"""
import pathlib, re
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
A_BODY = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
SEG = ('            CASE\n'
       "                WHEN hms > '090000' AND hms <= '093000' THEN 1\n"
       "                WHEN hms > '093000' AND hms <= '100000' THEN 2\n"
       "                WHEN hms > '100000' AND hms <= '103000' THEN 3\n"
       "                WHEN hms > '103000' AND hms <= '110000' THEN 4\n"
       "                WHEN hms > '110000' AND hms <= '113000' THEN 5\n"
       "                WHEN hms > '130000' AND hms <= '133000' THEN 6\n"
       "                WHEN hms > '133000' AND hms <= '140000' THEN 7\n"
       "                WHEN hms > '140000' AND hms <= '143000' THEN 8\n"
       "                WHEN hms > '143000' AND hms <= '150000' THEN 9\n"
       '                ELSE 0\n'
       '            END AS seg_i,\n')
UPSH_T = ('            (CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 '
          'THEN (px_high - GREATEST(px_open, px_close)) * seg_i ELSE 0 END) AS upsh_t_c,\n')
UTIME_CODE = ('    _g = df.groupby(["instrument", "day"], sort=False)\n'
              '    _cen = _g["sum_upsh_t"].transform("sum") / _g["sum_upsh"].transform("sum").where(_g["sum_upsh"].transform("sum") > 0)\n'
              '    df["c_utime"] = -(_cen - 5.0)\n')
def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:70], t.count(old)); return t.replace(old, new, 1)
def add_keep(s, cols, num):
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S); assert m, num
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body: body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]
def utime(s, num):
    s = repl(s, A_BODY, A_BODY + SEG + UPSH_T, num + "_snap")
    s = repl(s, A_WIN, '            SUM(upsh_t_c) AS sum_upsh_t,\n' + A_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, f.sum_upsh_t\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", "sum_upsh_t"]'), num + "_cast")
    s = repl(s, A_RV8, A_RV8 + UTIME_CODE, num + "_code")
    return add_keep(s, ["c_utime"], num)
def slot(s, base_cols, base_w, base_wl, end_col, wb, wl, num):
    s = repl(s, base_cols, base_cols.replace(']\n', ', "%s"]\n' % end_col), num + "_cols")
    fmt = lambda xs: ", ".join(["1.0"] + ["%.2f" % x for x in xs[1:]])
    s = repl(s, '    _W = np.tile(np.array([%s]), (len(result), 1))\n' % fmt(base_w),
             '    _W = np.tile(np.array([%s, %.2f]), (len(result), 1))\n' % (fmt(base_w), wb), num + "_w")
    s = repl(s, '    _W[_late] = np.array([%s])\n' % fmt(base_wl),
             '    _W[_late] = np.array([%s, %.2f])\n' % (fmt(base_wl), wl), num + "_wl")
    return s
def emit(num, name, s):
    p = D / ("%s_%s.py" % (num, name)); p.write_text(s, encoding="utf-8"); print("写出", p.name)

S823 = (D / "823_头部加隔夜跳空中性化二点零.py").read_text(encoding="utf-8")
S824 = (D / "824_头部加市值加权行业滞后中性化二点零.py").read_text(encoding="utf-8")
S816 = (D / "816_行业市值中性化加时点重心030.py").read_text(encoding="utf-8")
C481 = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_gap"]\n'
W481 = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15]), (len(result), 1))\n'
WL481 = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10])\n'
C482 = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indmwrev"]\n'
W482 = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10]), (len(result), 1))\n'
WL482 = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08])\n'

s = utime(S823, "838"); s = slot(s, C481, [1.0,0.30,0.15,0.20,0.15,0.15], [1.0,0.20,0.10,0.15,0.10,0.10], "c_utime", 0.45, 0.30, "838")
emit("838", "头部隔夜跳空加时点重心045", s)
s = utime(S824, "839"); s = slot(s, C482, [1.0,0.30,0.15,0.20,0.15,0.10], [1.0,0.20,0.10,0.15,0.10,0.08], "c_utime", 0.30, 0.20, "839")
emit("839", "头部市值加权行业滞后加时点重心030", s)
N20 = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'
emit("840", "行业市值中性化加时点重心span10平滑", repl(S816, N20,
     '    _os = result.sort_values(["instrument", "date"])\n'
     '    result["signal"] = _os.groupby("instrument", sort=False)["signal"].transform(\n'
     '        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)\n' + N20, "840"))
