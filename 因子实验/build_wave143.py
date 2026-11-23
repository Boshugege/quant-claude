# -*- coding: utf-8 -*-
"""第一百四十三批：按「每一个维度」分别优化，不看公榜估计/赋分。
维度榜首现状（评分机原始指标）：
  IC 均值/多空   -> 760（纯条件反转，IC 0.0795 / LS 46.69，但 IC_IR 1.30 / 压力 1.22 / 换手 0.59）
  IC_IR/压力     -> 489/433（重中性化，IC_IR 1.45-1.46 / 压力 1.37-1.38，但 IC 均值/多空偏低）
  换手           -> 头部 766 族（0.46，但 IC/多空低）
本批把「条件反转」这条高 IC/多空臂分别推 IC_IR、压力、换手三个方向，并把多分组条件反转合成。
- 809：760 + 2.0 倍中性化（IC_IR/压力臂）
- 810：760 + 2.5 倍中性化探针
- 811：760 + 时点重心分量（IC 均值/多空臂）
- 812：多分组条件反转合成纯源（IC 均值/多空臂）
- 813：760 信号 span5 平滑（换手臂）
- 814：812 合成 + 2.0 倍中性化（IC_IR/压力臂）
- 815：760 + 时点重心 + 2.0 倍（四维同时推）
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
S = (D / "760_自身换手历史五分位条件反转.py").read_text(encoding="utf-8")

A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
SIGNAL = '    result["signal"] = result["c_cond_turnself"]\n'
FACTOR = '    result["factor"] = result["signal"]\n'
UTIME_TAIL = '    df["c_utime"] = -(_cen - 5.0)\n'


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


def add_utime(s, num):
    seg = ('            CASE\n'
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
    upsh_case = ('CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 '
                 'THEN (px_high - GREATEST(px_open, px_close)) * seg_i ELSE 0 END')
    snap = seg + '            (%s) AS upsh_t_c,\n' % upsh_case
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, '            SUM(upsh_t_c) AS sum_upsh_t,\n' + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, f.sum_upsh_t\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", "sum_upsh_t"]'), num + "_cast")
    code = ('    _g = df.groupby(["instrument", "day"], sort=False)\n'
            '    _cen = _g["sum_upsh_t"].transform("sum") / _g["sum_upsh"].transform("sum").where(_g["sum_upsh"].transform("sum") > 0)\n'
            '    df["c_utime"] = -(_cen - 5.0)\n')
    s = repl(s, A_RV8, A_RV8 + code, num + "_code")
    return add_keep(s, ["c_utime"], num)


def neu(s, frac, num):
    return repl(s, FACTOR, '    result["factor"] = neutralize("signal", CHARS4, %.2f)\n' % frac, num + "_neu")


def emit(num, name, s):
    p = D / ("%s_%s.py" % (num, name)); p.write_text(s, encoding="utf-8"); print("写出", p.name)


# 809 / 810 中性化臂
emit("809", "条件反转中性化二点零", neu(S, 2.0, "809"))
emit("810", "条件反转中性化二点五探针", neu(S, 2.5, "810"))

# 811 时点重心合成（IC/多空臂）
s = add_utime(S, "811")
s = repl(s, SIGNAL, '    result["signal"] = rank_blend(["c_cond_turnself", "c_utime"], [1.0, 0.30])\n', "811_sig")
emit("811", "条件反转加时点重心", s)

# 812 多分组条件反转合成纯源（IC/多空臂）
s = repl(S, SIGNAL, '    result["signal"] = rank_blend(["c_cond_turnself", "c_cond_illiq", "c_cond_maxr", "c_cond_avgtr"], [1.0, 1.0, 1.0, 1.0])\n', "812_sig")
emit("812", "多分组条件反转合成纯源", s)

# 813 换手臂：信号 span5 平滑
s = repl(S, FACTOR,
         '    _ou = result.sort_values(["instrument", "date"])\n'
         '    result["signal"] = _ou.groupby("instrument", sort=False)["signal"].transform(\n'
         '        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(result.index)\n'
         '    result["factor"] = result["signal"]\n', "813_smooth")
emit("813", "条件反转span5平滑", s)

# 814 多分组合成 + 2.0 中性化（IC_IR/压力臂）
s = repl(S, SIGNAL, '    result["signal"] = rank_blend(["c_cond_turnself", "c_cond_illiq", "c_cond_maxr", "c_cond_avgtr"], [1.0, 1.0, 1.0, 1.0])\n', "814_sig")
emit("814", "多分组条件反转合成中性化二点零", neu(s, 2.0, "814"))

# 815 条件反转 + 时点重心 + 2.0 中性化（四维同推）
s = add_utime(S, "815")
s = repl(s, SIGNAL, '    result["signal"] = rank_blend(["c_cond_turnself", "c_utime"], [1.0, 0.30])\n', "815_sig")
emit("815", "条件反转加时点重心中性化二点零", neu(s, 2.0, "815"))
