# -*- coding: utf-8 -*-
"""第一百四十四批：把「压力/IC_IR 榜首 489（行业+前日市值中性化）」与「IC 均值/多空榜首 760/时点重心」两条臂继续对推。
- 816：489 + 时点重心 0.30（压力/IC_IR 臂上加 IC 均值）
- 817：489 + 2.5 倍中性化（压力臂）
- 818：789（头部+时点重心 0.30）时点重心权重提到 0.60
- 819：789 用时点重心 span5 平滑版 0.30（换手臂）
- 820：489 + 时点重心 0.30 + 2.5 倍
- 821：794（纯行业内市值加权同伴残差）信号 span5 平滑（换手臂）
- 822：811（条件反转+时点重心）+ 2.5 倍探针
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


def add_keep(s, cols, num):
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
    assert m, num + "_keep"
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


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


def emit(num, name, s):
    p = D / ("%s_%s.py" % (num, name)); p.write_text(s, encoding="utf-8"); print("写出", p.name)


# ---------- 489 基础：注入时点重心 ----------
S489 = (D / "489_自造特征加行业中性化前日市值强度二点零.py").read_text(encoding="utf-8")
A489_BODY = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A489_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A489_SEL = '        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n'
A489_CAST = '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n'
A489_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
SIG489 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
NEU489 = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'


def utime_489(s, num):
    s = repl(s, A489_BODY, A489_BODY + SEG + UPSH_T, num + "_snap")
    s = repl(s, A489_WIN, '            SUM(upsh_t_c) AS sum_upsh_t,\n' + A489_WIN, num + "_win")
    s = repl(s, A489_SEL, A489_SEL.replace("f.sum_deals\n", "f.sum_deals, f.sum_upsh_t\n"), num + "_sel")
    s = repl(s, A489_CAST, A489_CAST.replace('"sum_deals"]', '"sum_deals", "sum_upsh_t"]'), num + "_cast")
    s = repl(s, A489_RV8, A489_RV8 + UTIME_CODE, num + "_code")
    return add_keep(s, ["c_utime"], num)


s = utime_489(S489, "816")
s = repl(s, SIG489, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.30])\n', "816_sig")
emit("816", "行业市值中性化加时点重心030", s)

emit("817", "行业市值中性化二点五探针", repl(S489, NEU489, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "817"))

s = utime_489(S489, "820")
s = repl(s, SIG489, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.30])\n', "820_sig")
emit("820", "行业市值中性化加时点重心二点五", repl(s, NEU489, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "820_neu"))

# ---------- 789 基础 ----------
S789 = (D / "789_头部换手角加时点重心030.py").read_text(encoding="utf-8")
SIG789 = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670", "c_utime"],\n'
          '                                  [1.00, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20, 0.30])\n')
emit("818", "头部换手角时点重心060", repl(S789, SIG789, SIG789.replace("0.30])\n", "0.60])\n"), "818"))

UTIME_TAIL = '    df["c_utime"] = -(_cen - 5.0)\n'
smooth = ('    _ord_u = df.sort_values(["instrument", "date"])\n'
          '    df["c_utime_s"] = _ord_u.groupby("instrument", sort=False)["c_utime"].transform(\n'
          '        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(df.index)\n')
s = repl(S789, UTIME_TAIL, UTIME_TAIL + smooth, "819_smooth")
s = add_keep(s, ["c_utime_s"], "819")
s = repl(s, SIG789, SIG789.replace('"c_utime"]', '"c_utime_s"]'), "819_sig")
emit("819", "头部换手角平滑时点重心030", s)

# ---------- 794 平滑（换手臂） ----------
S794 = (D / "794_纯源行业内市值加权同伴残差两点零中性化.py").read_text(encoding="utf-8")
FACT = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'
s = repl(S794, FACT,
         '    _os = result.sort_values(["instrument", "date"])\n'
         '    result["signal"] = _os.groupby("instrument", sort=False)["signal"].transform(\n'
         '        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(result.index)\n' + FACT, "821_smooth")
emit("821", "行业内市值加权同伴残差span5平滑", s)

# ---------- 811 + 2.5 探针 ----------
S811 = (D / "811_条件反转加时点重心.py").read_text(encoding="utf-8")
F811 = '    result["factor"] = result["signal"]\n'
emit("822", "条件反转加时点重心二点五探针", repl(S811, F811, '    result["factor"] = neutralize("signal", CHARS4, 2.5)\n', "822"))
