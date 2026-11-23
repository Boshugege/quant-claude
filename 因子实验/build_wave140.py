# -*- coding: utf-8 -*-
"""第一百四十批：把内部得分第一的「上影线时点重心（c_utime）」轴搬进当前本地最高头部 766，
并做一批纯源/中性化确认。786/787 已在飞行中，本批补齐队列用。

- 788 / 789：766 + c_utime 0.15 / 0.30（头部 + 新分钟时点源）
- 790：766 中性化 2.25（2.0 与 2.5 之间的探针）
- 791：纯 c_utime 2.0 中性化（复核内部得分第一的源）
- 792：纯真换手率相对自身 2.0 中性化
- 793：纯行业滞后收益反转 2.0 中性化
- 794：纯行业内市值加权同伴残差 2.0 中性化
- 795：纯条件反转 2.0 中性化
- 796：766 把条件反转分量换成 c_utime 并加真换手分量（源替换实验）
- 797：纯 c_utime 加 span5 平滑 2.0 中性化（新源低换手版）
- 798：纯 c_utime + 真换手 + 行业滞后 合成 2.0 中性化（新源合成探针）
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE766 = (D / "766_头部换手角二点零中性化.py").read_text(encoding="utf-8")

A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
SIG = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"],\n'
       '                                  [1.0, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20])\n')
NEU = '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
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
    """注入 c_utime（上影线时点重心，日级）。"""
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
    win = '            SUM(upsh_t_c) AS sum_upsh_t,\n'
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    names = ("sum_upsh_t",)
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), num + "_cast")
    code = ('    # 上影线时点重心（日级）：Σ(时段序号 × 上影线) / Σ上影线，越大=冲高回落越晚。\n'
            '    _g = df.groupby(["instrument", "day"], sort=False)\n'
            '    _cen = _g["sum_upsh_t"].transform("sum") / _g["sum_upsh"].transform("sum").where(_g["sum_upsh"].transform("sum") > 0)\n'
            '    df["c_utime"] = -(_cen - 5.0)\n')
    s = repl(s, A_RV8, A_RV8 + code, num + "_code")
    return add_keep(s, ["c_utime"], num)


UTIME_TAIL = '    df["c_utime"] = -(_cen - 5.0)\n'


def add_utime_smooth(s, num):
    """再加一列 span5 平滑的 c_utime（低换手版），必须放在 c_utime 之后。"""
    code = ('    _ord_u = df.sort_values(["instrument", "date"])\n'
            '    df["c_utime_s"] = _ord_u.groupby("instrument", sort=False)["c_utime"].transform(\n'
            '        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(df.index)\n')
    s = repl(s, UTIME_TAIL, UTIME_TAIL + code, num + "_usmooth")
    return add_keep(s, ["c_utime_s"], num)


def set_signal(s, cols, weights, num):
    new = '    result["signal"] = rank_blend([%s],\n                                  [%s])\n' % (
        ", ".join('"%s"' % c for c in cols), ", ".join("%.2f" % w for w in weights))
    return repl(s, SIG, new, num + "_sig")


def set_neu(s, frac, num):
    return repl(s, NEU, '    result["factor"] = neutralize("signal", CHARS4, %.2f)\n' % frac, num + "_neu")


HEAD7 = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670"]
HEAD7W = [1.0, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20]

CANDS = []


def emit(num, name, s):
    s = s.replace("# -*- coding: utf-8 -*-\n", "# -*- coding: utf-8 -*-\n", 1)
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    CANDS.append(p)
    print("写出", p.name)


# 788/789 头部 + c_utime
for num, w in (("788", 0.15), ("789", 0.30)):
    s = add_utime(BASE766, num)
    s = set_signal(s, HEAD7 + ["c_utime"], HEAD7W + [w], num)
    emit(num, "头部换手角加时点重心%03d" % int(w * 100), s)

# 790 头部 2.25 中性化探针
s = set_neu(BASE766, 2.25, "790")
emit("790", "头部换手角两点二五中性化探针", s)

# 791 纯 c_utime 2.0
s = add_utime(BASE766, "791")
s = set_signal(s, ["c_utime"], [1.0], "791")
emit("791", "纯源时点重心两点零中性化", s)

# 797 纯 c_utime span5 平滑 2.0
s = add_utime(BASE766, "797")
s = add_utime_smooth(s, "797")
s = set_signal(s, ["c_utime_s"], [1.0], "797")
emit("797", "纯源时点重心平滑span5两点零中性化", s)

# 792/793/794/795 纯源
for num, col, name in (("792", "c_turnself", "纯源真换手相对自身两点零中性化"),
                       ("793", "c_indrev", "纯源行业滞后收益反转两点零中性化"),
                       ("794", "c_indpeer", "纯源行业内市值加权同伴残差两点零中性化"),
                       ("795", "c_condrev", "纯源条件反转两点零中性化")):
    s = set_signal(BASE766, [col], [1.0], num)
    emit(num, name, s)

# 796 头部：条件反转分量 -> c_utime，并补真换手分量
s = add_utime(BASE766, "796")
s = set_signal(s, ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_utime", "c_turnself"],
               [1.0, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20, 0.15], "796")
emit("796", "头部换手角时点重心置换条件反转", s)

# 798 纯源合成 c_utime + 真换手 + 行业滞后
s = add_utime(BASE766, "798")
s = set_signal(s, ["c_utime", "c_turnself", "c_indrev"], [1.0, 1.0, 1.0], "798")
emit("798", "纯源时点重心真换手行业滞后合成两点零", s)

print("共写出 %d 个因子" % len(CANDS))
