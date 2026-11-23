# -*- coding: utf-8 -*-
"""第五十八批（405+）：继续上全新方向（底座 373）。

- 405 分钟内收盘位置均值：(收-低)/(高-低) 的窗口均值——分钟级买盘压制强度。
- 406 分钟收益一阶自相关：窗口内 Σ(r1·r1_lag)/Σr1²（唯一允许的单列 LAG）。
- 407 上影线 80 窗口形态（对 40/EWMA10 的时窗再扫一个点）。
- 408 隔夜跳空分量：log(昨收 / 前一交易日末价)。
- 409 隔夜跳空的同伴残差。
- 410 上午/下午成交额比。
- 412 最大单分钟波动占已实现方差比（跳跃集中度）。
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
A_LADDER = '    result["ladder"] = peer_ladder()\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'


def transparent(s, num, snap, win, names):
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), num + "_cast")
    return s


def add_sig(s, num, col, w):
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "%s"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % (col, w))
    return repl(s, SIG, sig, num + "_sig")


def c405(s):
    s = transparent(s, "405",
                    '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low THEN (px_close - px_low) / NULLIF(px_high - px_low, 0) ELSE 0 END AS pos_c,\n',
                    '            AVG(pos_c) AS avg_pos,\n', ("avg_pos",))
    code = ('    # 分钟内收盘位置均值：(收-低)/(高-低) 的窗口均值，40 窗口均值。\n'
            '    df["c_pos"] = roll(df["avg_pos"], 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "405_code")
    s = add_keep(s, ["c_pos"], "405")
    return add_sig(s, "405", "c_pos", 0.15)


def c406(s):
    """分钟收益一阶自相关：把 cte_snapshot 扩一列 LAG，改 cte_window 的来源。"""
    s = repl(s, '    cte_window AS (\n        SELECT\n',
             '    cte_lag AS (\n        SELECT *, LAG(r1_c) OVER (PARTITION BY instrument_id, trading_day, time_segment ORDER BY hms) AS r1_lag\n        FROM cte_snapshot\n    ),\n    cte_window AS (\n        SELECT\n', "406_cte")
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + '            CASE WHEN px_open > 0 THEN (px_close - px_open) / px_open ELSE 0 END AS r1_c,\n', "406_snap")
    s = repl(s, A_DEALS_WIN, '            SUM(r1_c * r1_lag) AS sum_r1lag,\n            SUM(r1_c * r1_c) AS sum_r1sq2,\n' + A_DEALS_WIN, "406_win")
    s = repl(s, '        FROM cte_snapshot\n        WHERE time_segment != -1\n', '        FROM cte_lag\n        WHERE time_segment != -1\n', "406_from")
    names = ("sum_r1lag", "sum_r1sq2")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), "406_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), "406_cast")
    s = add_keep(s, list(names) + ["c_mar"], "406")
    code = ('    # 分钟收益一阶自相关：窗口内 Σ(r1·r1_lag)/Σr1²，40 窗口均值。\n'
            '    df["c_mar"] = roll(df["sum_r1lag"] / df["sum_r1sq2"].where(df["sum_r1sq2"] > 0), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "406_code")
    return add_sig(s, "406", "c_mar", 0.15)


def c407(s):
    new = '    _upsh2 = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_upsh"] = -roll(_upsh2, 80, "mean", 20)\n'
    return repl(s, A_UPSH, new, "407_win80")


def c408(s):
    """隔夜跳空：log(昨收 / 前一交易日末价)。"""
    code = ('    # 隔夜跳空：log(昨收 / 前一交易日末价)，40 窗口均值。\n'
            '    _dlast = df.groupby(["instrument", "day"], sort=False)["last_px"].last()\n'
            '    _dlast.name = "dlast"\n'
            '    _df2 = df.merge(_dlast.reset_index(), on=["instrument", "day"], how="left")\n'
            '    _prev = _df2.groupby("instrument", sort=False)["dlast"].shift(1)\n'
            '    _gap = np.log(df["pre_close"] / _prev.where(_prev > 0))\n'
            '    df["c_gap"] = roll(_gap, 40, "mean", 10)\n')
    s = repl(s, A_RV8, A_RV8 + code, "408_code")
    s = add_keep(s, ["c_gap"], "408")
    return add_sig(s, "408", "c_gap", 0.15)


def c409(s):
    """隔夜跳空的同伴残差。"""
    s = c408(s)
    resid = '    result["c_gap_p"] = result["c_gap"] - peer_mean("c_gap").fillna(result.groupby("date")["c_gap"].transform("mean"))\n'
    s = repl(s, A_LADDER, A_LADDER + resid, "409_resid")
    s = s.replace('"c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15]', '"c_gap_p"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.20]', 1)
    return s


def c410(s):
    """上午/下午成交额比。"""
    code = ('    df["hhmm"] = df["date"].dt.strftime("%H%M").astype(int)\n'
            '    _amv = df["sum_amt"].where(df["hhmm"] <= 1130, 0.0)\n'
            '    _pmv = df["sum_amt"].where(df["hhmm"] >= 1330, 0.0)\n'
            '    _a = df.groupby(["instrument", "day"], sort=False)["_amv"].transform("sum") if False else _amv.groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
            '    _p = _pmv.groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
            '    _ratio = np.log((_p + 1.0) / (_a + 1.0))\n'
            '    df["c_amshape"] = roll(_ratio, 40, "mean", 10)\n')
    s = repl(s, A_RV8, A_RV8 + code, "410_code")
    s = add_keep(s, ["c_amshape"], "410")
    return add_sig(s, "410", "c_amshape", 0.15)


def c412(s):
    s = transparent(s, "412",
                    '            CASE WHEN px_open > 0 THEN ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) ELSE 0 END AS r1sq_c,\n',
                    '            MAX(r1sq_c) AS max_r1sq,\n            SUM(r1sq_c) AS sum_r1sq,\n', ("max_r1sq", "sum_r1sq"))
    code = ('    # 最大单分钟波动占已实现方差比（跳跃集中度），40 窗口均值取负。\n'
            '    _c = df["max_r1sq"] / df["sum_r1sq"].where(df["sum_r1sq"] > 0)\n'
            '    df["c_jumpsq"] = -roll(_c.where(lambda v: v > 0), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "412_code")
    s = add_keep(s, ["c_jumpsq"], "412")
    return add_sig(s, "412", "c_jumpsq", 0.15)


CANDS = [("405", "加分钟内收盘位置", c405),
         ("406", "加分钟收益自相关", c406),
         ("407", "上影线窗口八十", c407),
         ("408", "加隔夜跳空", c408),
         ("409", "加同伴残差隔夜跳空", c409),
         ("410", "加上下午成交额比", c410),
         ("411", "加跳跃集中度", c412)]

for num, name, fn in CANDS:
    s = fn(BASE373.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
