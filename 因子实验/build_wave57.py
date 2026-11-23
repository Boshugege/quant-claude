# -*- coding: utf-8 -*-
"""第五十七批（397+）：四个**全新**分钟级微观结构轴（此前任何家族都没做过）+ 上影线时窗扫描。

- 399 分钟量价相关：窗口内 corr(分钟收益, 分钟成交额)。
- 400 上涨分钟强度：Σ max(0, r1) / Σ|r1|（买盘持续性 / 分钟级不平衡）。
- 401 成交额方向加权：Σ sign(收-开)×额 / Σ额（不走 tick 的主动净买入代理）。
- 402 价格时间重心：Σ(分钟内时刻 × |r1|) / Σ|r1|（价格活动的日内时间重心，与"成交额时间重心"不同）。
- 397/398/403/404 为上影线 EWMA 时窗与其余组合的扫描。
底座：373（合法最好）。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE373 = D / "362_上影线EWMA十.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
    return t.replace(old, new, 1)


def add_keep(s, cols, num):
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
    assert m, num
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
A_UPSH = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'

MIN_SNAP = ('            CASE WHEN px_open > 0 THEN (px_close - px_open) / px_open ELSE 0 END AS r1_c,\n'
            '            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN amt ELSE 0 END AS amt_c,\n'
            '            CASE WHEN px_open > 0 AND px_close > px_open THEN (px_close - px_open) / px_open ELSE 0 END AS posr1_c,\n'
            '            CASE WHEN px_open > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr1_c,\n'
            '            CASE WHEN px_open > 0 THEN SIGN(px_close - px_open) * amt ELSE 0 END AS sbamt_c,\n'
            '            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr0_c,\n'
            '            (CAST(SUBSTR(hms, 1, 2) AS INTEGER) * 60 + CAST(SUBSTR(hms, 3, 2) AS INTEGER)) AS tpos_c,\n')
MIN_WIN = ('            SUM(r1_c) AS sum_r1,\n'
           '            SUM(r1_c * r1_c) AS sum_r1sq,\n'
           '            SUM(amt_c) AS sum_amtc,\n'
           '            SUM(amt_c * amt_c) AS sum_amt2c,\n'
           '            SUM(r1_c * amt_c) AS sum_r1amt,\n'
           '            SUM(posr1_c) AS sum_posr1,\n'
           '            SUM(absr0_c) AS sum_absr1,\n'
           '            SUM(sbamt_c) AS sum_sbamt,\n'
           '            SUM(tpos_c * absr0_c) AS sum_tabsr,\n'
           '            COUNT(*) AS n_min,\n')
MIN_NAMES = ("sum_r1", "sum_r1sq", "sum_amtc", "sum_amt2c", "sum_r1amt", "sum_posr1",
             "sum_absr1", "sum_sbamt", "sum_tabsr", "n_min")


def add_minute_block(s, num):
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + MIN_SNAP, num + "_snap")
    s = repl(s, A_DEALS_WIN, MIN_WIN + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in MIN_NAMES) + "\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in MIN_NAMES) + "]"), num + "_cast")
    return add_keep(s, list(MIN_NAMES), num)


def add_sig(s, num, col, w):
    s = add_keep(s, [col], num + "_keepcol")
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "%s"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % (col, w))
    return repl(s, SIG, sig, num + "_sig")


def c399(s):
    s = add_minute_block(s, "399")
    code = ('    # 分钟量价相关：窗口内 corr(分钟收益, 分钟成交额)，40 窗口均值。\n'
            '    _n = df["n_min"]\n'
            '    _cov = (_n * df["sum_r1amt"] - df["sum_r1"] * df["sum_amtc"]) / _n\n'
            '    _vx = (_n * df["sum_r1sq"] - df["sum_r1"] ** 2) / _n\n'
            '    _vy = (_n * df["sum_amt2c"] - df["sum_amtc"] ** 2) / _n\n'
            '    _c = _cov / np.sqrt((_vx * _vy).where(lambda v: v > 0))\n'
            '    df["c_rvcorr"] = roll(_c.clip(-1, 1), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "399_code")
    return add_sig(s, "399", "c_rvcorr", 0.15)


def c400(s):
    s = add_minute_block(s, "400")
    code = ('    # 上涨分钟强度：窗口内 Σmax(0,r1)/Σ|r1|，40 窗口均值。\n'
            '    _p = df["sum_posr1"] / df["sum_absr1"].where(df["sum_absr1"] > 0)\n'
            '    df["c_posmin"] = roll(_p, 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "400_code")
    return add_sig(s, "400", "c_posmin", 0.15)


def c401(s):
    s = add_minute_block(s, "401")
    code = ('    # 成交额方向加权：窗口内 Σ sign(收-开)×额 / Σ额，40 窗口均值。\n'
            '    _d = df["sum_sbamt"] / df["sum_amtc"].where(df["sum_amtc"] > 0)\n'
            '    df["c_amtdir"] = roll(_d, 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "401_code")
    return add_sig(s, "401", "c_amtdir", 0.15)


def c402(s):
    s = add_minute_block(s, "402")
    code = ('    # 价格时间重心：窗口内 Σ(时刻×|r1|)/Σ|r1|，40 窗口均值取负。\n'
            '    _tc = df["sum_tabsr"] / df["sum_absr1"].where(df["sum_absr1"] > 0)\n'
            '    df["c_ptc"] = -roll(_tc, 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "402_code")
    return add_sig(s, "402", "c_ptc", 0.15)


def span(s, sp, num):
    old = 'x.ewm(span=10, min_periods=8)'
    new = 'x.ewm(span=%d, min_periods=8)' % sp
    return repl(s, old, new, num + "_span")


def c403(s):
    """跳跃性用 EWMA(span10) 聚合而不是 40 窗口均值（与上影线一致的时间聚合）。"""
    snap = '            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr1_c,\n'
    win = '            AVG(absr1_c) AS avg_absr,\n            MAX(absr1_c) AS max_absr,\n'
    names = ("avg_absr", "max_absr")
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, "403_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, "403_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), "403_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), "403_cast")
    s = add_keep(s, list(names) + ["c_jump"], "403")
    code = ('    _jmp = df["max_absr"] / df["avg_absr"].where(df["avg_absr"] > 0)\n'
            '    _jmp = _jmp.where(lambda v: (v > 0) & np.isfinite(v))\n'
            '    df["c_jump"] = -_jmp.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "403_code")
    return add_sig(s, "403", "c_jump", 0.10)


def c404(s):
    """同时段异常放量（第五十四批 370 的合法版本，加到 373 上）。"""
    code = ('    df["slot"] = df["date"].dt.strftime("%H%M")\n'
            '    _base = df.groupby(["instrument", "slot"], sort=False)["sum_vol"].transform(\n'
            '        lambda x: x.rolling(20, min_periods=8).mean().shift(1))\n'
            '    df["c_absvol"] = df["sum_vol"] / _base.where(_base > 0)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "404_code")
    s = add_keep(s, ["sum_vol", "c_absvol"], "404")
    return add_sig(s, "404", "c_absvol", 0.15)


CANDS = [("397", "上影线EWMA六", lambda s: span(s, 6, "397")),
         ("398", "上影线EWMA十五", lambda s: span(s, 15, "398")),
         ("399", "分钟量价相关", c399),
         ("400", "上涨分钟强度", c400),
         ("401", "成交额方向加权", c401),
         ("402", "价格时间重心", c402),
         ("403", "上影线EWMA加跳跃性EWMA", c403),
         ("404", "上影线EWMA加同时段异常放量", c404)]

for num, name, fn in CANDS:
    s = fn(BASE373.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
