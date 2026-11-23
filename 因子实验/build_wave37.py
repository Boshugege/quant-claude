# -*- coding: utf-8 -*-
"""第四十五批（301+）：换新数据维度——路径/波动结构类 + 结构不同的同伴定义。

本族（283-296）已经收敛，本轮不再在现有轴上微调，改试**全新数据维度**：
- 301：日内 30 分钟收益的一阶自相关（trendiness，近 40 窗口）
- 302：窗口内**成交额加权价格离散度**（Σamt·px²/Σamt − (Σamt·px/Σamt)²，近 40 窗口）
- 303：日内低于当日 VWAP 的 30 分钟段占比（路径位置，近 40 窗口）
- 304：窗口高低幅波动 ln(高/低)²（近 40 窗口均值，作 0.10 小权重分量）
- 305：同伴定义换成**日内 30 分钟收益相关**（145 结构：PEER_LOOKBACK=160 / MIN_OBS=60）——结构不同的同伴轴
- 306：293 的实体占比权重 0.15→0.25（权重灵敏度）
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "293_规模加权同伴幂二EWMA十上影线加平均单笔二日尺度与实体占比.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


A_BODY_SNAP = ('            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n')
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n'
A_CBODY = '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
A_KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
A_PEER_PARAMS = '    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 240, 120, 2.0\n'
A_DAILYRET = ('    daily = df.groupby(["day", "instrument"], sort=False).agg(px_last=("last_px", "last"), pc=("pre_close", "first"))\n'
              '    daily_ret = safe_log_ratio(daily["px_last"], daily["pc"]).unstack("instrument").sort_index()\n')
A_RET30 = '    ret30 = safe_log_ratio(px, grp["last_px"].shift(1).where(same_day)).fillna(safe_log_ratio(px, df["pre_close"]))\n'

PX_EXPR = "(CASE WHEN bp1 > 0 AND ap1 > bp1 THEN (bp1 + ap1) / 2.0 WHEN px_close > 0 THEN px_close ELSE 0 END)"


def add_comp(s, comp_sql, win_sql, win_cols, sign_cols, comp_code, weight, num):
    """加 SQL 透传（win_cols）+ Python 分量（sign_cols 进 keep/signal）。"""
    if comp_sql:
        s = repl(s, A_BODY_SNAP, A_BODY_SNAP + comp_sql, num + "_snap")
    if win_sql:
        s = repl(s, A_DEALS_WIN, win_sql + A_DEALS_WIN, num + "_win")
    if win_cols:
        s = repl(s, A_SEL, A_SEL.rstrip("\n") + ", " + ", ".join("f.%s" % c for c in win_cols) + "\n", num + "_sel")
        cast_new = '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals", %s]:\n' % ", ".join('"%s"' % c for c in win_cols)
        s = repl(s, A_CAST, cast_new, num + "_cast")
    if comp_code:
        s = repl(s, A_CBODY, A_CBODY + comp_code, num + "_comp")
    new_keep = A_KEEP.rstrip("\n").replace('"c_body"]', '"c_body", %s]' % ", ".join('"%s"' % c for c in sign_cols)) + "\n"
    s = repl(s, A_KEEP, new_keep, num + "_keep")
    cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"] + list(sign_cols)
    wts = [1.0, 0.30, 0.15, 0.20, 0.15] + [weight] * len(sign_cols)
    sig = '    result["signal"] = rank_blend([%s], [%s])\n' % (
        ", ".join('"%s"' % c for c in cols), ", ".join("%.2f" % w for w in wts))
    s = repl(s, A_SIG, sig, num + "_sig")
    return s


def c301(s):
    code = (
        '    # 日内 30 分钟收益的一阶自相关（trendiness）：同日内相邻段收益的滚动相关系数（40 窗口），取负。\n'
        '    _sr = safe_log_ratio(px, grp["last_px"].shift(1)).where(same_day)\n'
        '    _prev = _sr.groupby(df["instrument"], sort=False).shift(1)\n'
        '    _pair = same_day & _sr.notna() & _prev.notna()\n'
        '    _x, _y = _prev.where(_pair), _sr.where(_pair)\n'
        '    _mx, _my = roll(_x, 40, "mean", 10), roll(_y, 40, "mean", 10)\n'
        '    _mx2, _my2, _mxy = roll(_x * _x, 40, "mean", 10), roll(_y * _y, 40, "mean", 10), roll(_x * _y, 40, "mean", 10)\n'
        '    _vx = (_mx2 - _mx * _mx).where(lambda v: v > 1e-12)\n'
        '    _vy = (_my2 - _my * _my).where(lambda v: v > 1e-12)\n'
        '    _ac = ((_mxy - _mx * _my) / np.sqrt(_vx * _vy)).clip(-1, 1)\n'
        '    df["c_trend"] = -_ac\n')
    s = repl(s, A_CBODY, A_CBODY + code, "301_comp")
    s = repl(s, A_KEEP, A_KEEP.rstrip("\n").replace('"c_body"]', '"c_body", "c_trend"]') + "\n", "301_keep")
    s = repl(s, A_SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_trend"], [1.00, 0.30, 0.15, 0.20, 0.15, 0.15])\n', "301_sig")
    return s


def c302(s):
    comp_sql = ('            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 '
                'THEN amt * %s ELSE 0 END AS amtpx_c,\n' % PX_EXPR)
    comp_sql += ('            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 '
                 'THEN amt * %s * %s ELSE 0 END AS amtpx2_c,\n' % (PX_EXPR, PX_EXPR))
    win_sql = ('            SUM(amtpx_c) AS sum_amtpx,\n'
               '            SUM(amtpx2_c) AS sum_amtpx2,\n')
    code = ('    # 窗口内成交额加权价格离散度：Σamt·px²/Σamt − (Σamt·px/Σamt)²，近 40 窗口均值取负。\n'
            '    _m1 = df["sum_amtpx"] / df["sum_amt"].where(df["sum_amt"] > 0)\n'
            '    _m2 = df["sum_amtpx2"] / df["sum_amt"].where(df["sum_amt"] > 0)\n'
            '    df["c_disp"] = -roll((_m2 - _m1 * _m1).where(lambda v: v > 0), 40, "mean", 10)\n')
    return add_comp(s, comp_sql, win_sql, ["sum_amtpx", "sum_amtpx2"], ["c_disp"], code, 0.15, "302")


def c303(s):
    code = ('    # 日内低于当日 VWAP 的 30 分钟段占比（路径位置），近 40 窗口均值取负。\n'
            '    _below = ((px > 0) & day_vwap.notna() & (px < day_vwap)).astype(float)\n'
            '    df["c_below"] = -roll(_below, 40, "mean", 10)\n')
    return add_comp(s, None, None, [], ["c_below"], code, 0.15, "303")


def c304(s):
    code = ('    # 窗口高低幅波动 ln(高/低)² 的近 40 窗口均值（取负，与低波动方向一致）。\n'
            '    _pk = (np.log(df["win_high"] / df["win_low"].where(df["win_low"] > 0)).where(df["win_low"] > 0)) ** 2\n'
            '    df["c_pk"] = -roll(_pk, 40, "mean", 10)\n')
    return add_comp(s, None, None, [], ["c_pk"], code, 0.10, "304")


def c305(s):
    s = repl(s, A_PEER_PARAMS, '    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 160, 60, 2.0\n', "305_params")
    s = repl(s, A_DAILYRET,
             '    daily_ret = ret30.to_frame("r").assign(\n'
             '        instrument=df["instrument"].values, date=df["date"].values\n'
             '    ).pivot(index="date", columns="instrument", values="r").sort_index()\n', "305_dailyret")
    return s


def c306(s):
    s = repl(s, A_SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.00, 0.30, 0.15, 0.20, 0.25])\n', "306_sig")
    return s


CANDS = [("301", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与段收益自相关", c301),
         ("302", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与价格离散度", c302),
         ("303", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与低于VWAP段占比", c303),
         ("304", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与高低幅波动", c304),
         ("305", "日内相关同伴加实体平均单笔二日尺度", c305),
         ("306", "实体占比二成五", c306)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
