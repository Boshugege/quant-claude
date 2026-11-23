# -*- coding: utf-8 -*-
"""第一百四十六批：多空夏普臂（481/482）与时点重心/中性化合并，冲击「五维同时靠前」。
- 828：482（头部+市值加权行业滞后，LS 榜首族）+ 时点重心 0.30 + 2.0 倍
- 829：481（头部+隔夜跳空）+ 时点重心 0.30 + 2.0 倍
- 830：482 + 时点重心 0.30（1.5 倍，保多空）
- 831：481 + 时点重心 0.30（1.5 倍）
- 832：481 同时加隔夜跳空与市值加权行业滞后两分量 + 2.0 倍
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
    assert t.count(old) == 1, (tag, old[:70], t.count(old))
    return t.replace(old, new, 1)


def add_keep(s, cols, num):
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S); assert m, num
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


def utime(s, num):
    s = repl(s, A_BODY, A_BODY + SEG + UPSH_T, num + "_snap")
    s = repl(s, A_WIN, '            SUM(upsh_t_c) AS sum_upsh_t,\n' + A_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, f.sum_upsh_t\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", "sum_upsh_t"]'), num + "_cast")
    s = repl(s, A_RV8, A_RV8 + UTIME_CODE, num + "_code")
    return add_keep(s, ["c_utime"], num)


def slot_add(s, base_cols, base_w, base_wl, wb, wl, num):
    s = repl(s, base_cols, base_cols.replace(']\n', ', "c_utime"]\n'), num + "_cols")
    fmt = lambda xs: ", ".join(["1.0"] + ["%.2f" % x for x in xs[1:]])
    aw = '    _W = np.tile(np.array([%s]), (len(result), 1))\n' % fmt(base_w)
    neww = '    _W = np.tile(np.array([%s, %.2f]), (len(result), 1))\n' % (fmt(base_w), wb)
    s = repl(s, aw, neww, num + "_w")
    awl = '    _W[_late] = np.array([%s])\n' % fmt(base_wl)
    newwl = '    _W[_late] = np.array([%s, %.2f])\n' % (fmt(base_wl), wl)
    s = repl(s, awl, newwl, num + "_wl")
    return s


def emit(num, name, s):
    p = D / ("%s_%s.py" % (num, name)); p.write_text(s, encoding="utf-8"); print("写出", p.name)


S481 = (D / "481_头部加隔夜跳空.py").read_text(encoding="utf-8")
S482 = (D / "482_头部加市值加权行业滞后.py").read_text(encoding="utf-8")
C481 = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_gap"]\n'
W481 = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15]), (len(result), 1))\n'
WL481 = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10])\n'
C482 = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indmwrev"]\n'
W482 = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10]), (len(result), 1))\n'
WL482 = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08])\n'
N15 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
N20 = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'

# 828: 482 + utime + 2.0
s = utime(S482, "828"); s = slot_add(s, C482, [1.0, 0.30, 0.15, 0.20, 0.15, 0.10], [1.0, 0.20, 0.10, 0.15, 0.10, 0.08], 0.30, 0.20, "828")
s = repl(s, N15, N20, "828_neu"); emit("828", "市值加权行业滞后加时点重心二点零", s)

# 829: 481 + utime + 2.0
s = utime(S481, "829"); s = slot_add(s, C481, [1.0, 0.30, 0.15, 0.20, 0.15, 0.15], [1.0, 0.20, 0.10, 0.15, 0.10, 0.10], 0.30, 0.20, "829")
s = repl(s, N15, N20, "829_neu"); emit("829", "隔夜跳空加时点重心二点零", s)

# 830: 482 + utime 1.5
s = utime(S482, "830"); s = slot_add(s, C482, [1.0, 0.30, 0.15, 0.20, 0.15, 0.10], [1.0, 0.20, 0.10, 0.15, 0.10, 0.08], 0.30, 0.20, "830")
emit("830", "市值加权行业滞后加时重心一点五", s)

# 831: 481 + utime 1.5
s = utime(S481, "831"); s = slot_add(s, C481, [1.0, 0.30, 0.15, 0.20, 0.15, 0.15], [1.0, 0.20, 0.10, 0.15, 0.10, 0.10], 0.30, 0.20, "831")
emit("831", "隔夜跳空加时点重心一点五", s)

# 832: 481 同时加隔夜跳空与市值加权滞后 + 2.0
s = S481
s = repl(s, C481, '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_gap", "c_indmwrev"]\n', "832_cols")
s = repl(s, W481, '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15, 0.10]), (len(result), 1))\n', "832_w")
s = repl(s, WL481, '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10, 0.08])\n', "832_wl")
s = repl(s, N15, N20, "832_neu"); emit("832", "隔夜跳空与市值加权行业滞后双分量二点零", s)
