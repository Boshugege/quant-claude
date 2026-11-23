# -*- coding: utf-8 -*-
"""第五十六批（389+）：以 373（= 353 + 上影线 EWMA span10，当前合法最好，IC_IR/压力/换手全面优于 258）为底座叠加。

373 与 258 直接配对：综合 +0.0203(z=1.40)，IC z=+5.3、换手 z=−26.9 显著更好，IC_IR/压力不显著变差且已在参照池第一。
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
A_NEU = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
A_DN = '    df["c_dnsh"] = -roll(df["sum_dnsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'

A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'


def add_sig(s, num, col, w):
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "%s"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % (col, w))
    return repl(s, SIG, sig, num + "_sig")


def minute_col(s, num, snap, win, names):
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), num + "_cast")
    return s


JUMP_SNAP = '            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr1_c,\n'
JUMP_WIN = '            AVG(absr1_c) AS avg_absr,\n            MAX(absr1_c) AS max_absr,\n'
RESID = '    result["c_jump_p"] = result["c_jump"] - peer_mean("c_jump").fillna(result.groupby("date")["c_jump"].transform("mean"))\n'


def add_jump_resid(s, num, w):
    s = minute_col(s, num, JUMP_SNAP, JUMP_WIN, ("avg_absr", "max_absr"))
    code = ('    _jmp = df["max_absr"] / df["avg_absr"].where(df["avg_absr"] > 0)\n'
            '    _jmp = _jmp.where(lambda v: (v > 0) & np.isfinite(v))\n'
            '    df["c_jump"] = -roll(_jmp, 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, num + "_jump")
    s = add_keep(s, ["c_jump"], num)
    s = repl(s, A_LADDER, A_LADDER + RESID, num + "_resid")
    return add_sig(s, num, "c_jump_p", w)


def slot_weights(s, num):
    helper = ('    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
              '    _rk = (result.groupby("date")[_cols].rank(pct=True) - 0.5).to_numpy()\n'
              '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15]), (len(result), 1))\n'
              '    _late = result["date"].dt.strftime("%H%M").isin(["1400", "1430"]).to_numpy()\n'
              '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10])\n'
              '    _has = ~np.isnan(_rk)\n'
              '    _den = (_has * _W).sum(axis=1)\n'
              '    _num = np.nansum(np.nan_to_num(_rk) * _W, axis=1)\n'
              '    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)\n')
    return repl(s, SIG, helper, num + "_slotw")


def c389(s):
    return add_jump_resid(s, "389", 0.10)
def c390(s):
    return slot_weights(s, "390")
def c391(s):
    s = repl(s, A_UPSH, A_UPSH + A_DN, "391_code")
    s = add_keep(s, ["c_dnsh"], "391")
    return add_sig(s, "391", "c_dnsh", 0.15)
def c392(s):
    return add_jump_resid(s, "392", 0.05)
def c393(s):
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.30, 0.15])\n'
    return repl(s, SIG, sig, "393_sig")
def c394(s):
    s = minute_col(s, "394",
                   '            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN amt ELSE 0 END AS amt_c,\n',
                   '            MAX(amt_c) AS max_amt,\n', ("max_amt",))
    code = ('    _mx = df["max_amt"] / df["sum_amt"].where(df["sum_amt"] > 0)\n'
            '    df["c_maxshare"] = -roll(_mx.where(lambda v: v > 0), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "394_code")
    s = add_keep(s, ["c_maxshare"], "394")
    return add_sig(s, "394", "c_maxshare", 0.15)
def c395(s):
    return repl(s, A_NEU, '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n', "395_neu")
def c396(s):
    new = ('    result["sig_a"] = rank_blend(["ladder", "c_upsh", "c_avgtr"], [1.0, 0.30, 0.15])\n'
           '    result["sig_b"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
           '    result["signal"] = 0.5 * (result.groupby("date")["sig_a"].rank(pct=True) - 0.5) + 0.5 * (result.groupby("date")["sig_b"].rank(pct=True) - 0.5)\n')
    return repl(s, SIG, new, "396_blend")


CANDS = [("389", "上影线EWMA加同伴残差跳跃性010", c389),
         ("390", "上影线EWMA加分槽位权重", c390),
         ("391", "上影线EWMA加下影线", c391),
         ("392", "上影线EWMA加同伴残差跳跃性005", c392),
         ("393", "上影线EWMA两日尺度权重030", c393),
         ("394", "上影线EWMA加分钟最大额占比", c394),
         ("395", "上影线EWMA中性化强度20", c395),
         ("396", "上影线EWMA与258信号混合50", c396)]

for num, name, fn in CANDS:
    s = fn(BASE373.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
