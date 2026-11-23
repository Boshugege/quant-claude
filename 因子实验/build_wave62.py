# -*- coding: utf-8 -*-
"""第六十二批（437+）：又四条新信息源 + 三个同伴残差方向（底座 373）。

- 437 报价深度/成交额比（挂单厚度相对当日换手，纯订单簿）；
- 438 盘口不平衡的窗口内波动（挂单结构的不稳定性）；
- 439 量价弹性（Kyle lambda：窗口内 cov(r, v) / var(v)）；
- 440 分钟收益已实现峰度（Σr⁴/Σr²²）；
- 441/442/443 同伴残差化：同伴上一时段收益（lag2）、上影线占比、r_rv16（无 SQL 变化）。
437–440 共用同一份 SQL 文本（同批只算一次 SQL）。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
B373 = D / "362_上影线EWMA十.py"

A_UPSH = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'
A_LADDER = '    result["ladder"] = peer_ladder()\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_RAW = '            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'
SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'


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


def add_sig(s, num, col, w):
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "%s"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % (col, w))
    return repl(s, SIG, sig, num + "_sig")


def transparent(s, num, snap, win, names):
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), num + "_cast")
    return s


BID5 = "(bv1 + bv2 + bv3 + bv4 + bv5)"
ASK5 = "(av1 + av2 + av3 + av4 + av5)"
R1 = "CASE WHEN px_open > 0 THEN (px_close - px_open) / px_open ELSE 0 END"

OB_RAW = (''.join('            CAST(COALESCE(%s, 0) AS DOUBLE) AS %s,\n' % (c, a)
                  for c, a in [("bid_volume1", "bv1"), ("bid_volume2", "bv2"), ("bid_volume3", "bv3"),
                               ("bid_volume4", "bv4"), ("bid_volume5", "bv5"), ("ask_volume1", "av1"),
                               ("ask_volume2", "av2"), ("ask_volume3", "av3"), ("ask_volume4", "av4"),
                               ("ask_volume5", "av5")]))
RICH_SNAP = (
    '            %s AS bidv_c,\n' % BID5 +
    '            %s AS askv_c,\n' % ASK5 +
    '            (%s - %s) / (%s + %s + 1.0) AS ob_c,\n' % (BID5, ASK5, BID5, ASK5) +
    '            %s AS r1_c,\n' % R1 +
    '            (%s) * (%s) AS r1sq_c,\n' % (R1, R1) +
    '            (%s) * (%s) * (%s) * (%s) AS r1q_c,\n' % (R1, R1, R1, R1) +
    '            vol_c * (%s) AS rv_c,\n' % R1 +
    '            vol_c * vol_c AS v2_c,\n')
RICH_WIN = (
    '            SUM(bidv_c) AS sum_bidv,\n'
    '            SUM(askv_c) AS sum_askv,\n'
    '            SUM(ob_c) AS sum_ob,\n'
    '            SUM(ob_c * ob_c) AS sum_ob2,\n'
    '            SUM(r1_c) AS sum_r1,\n'
    '            SUM(r1sq_c) AS sum_r1sq,\n'
    '            SUM(r1q_c) AS sum_r1q,\n'
    '            SUM(rv_c) AS sum_rv,\n'
    '            SUM(v2_c) AS sum_v2,\n')
RICH_NAMES = ("sum_bidv", "sum_askv", "sum_ob", "sum_ob2", "sum_r1", "sum_r1sq", "sum_r1q", "sum_rv", "sum_v2")


def rich(s, num):
    s = repl(s, A_RAW, A_RAW[:-1] + ",\n" + OB_RAW, num + "_raw")
    return transparent(s, num, RICH_SNAP, RICH_WIN, RICH_NAMES)


def c437(s):
    """报价深度/成交额比。"""
    s = rich(s, "437")
    code = ('    # 报价深度/成交额比：窗口内 (五档买+卖挂单量) / 成交额，EWMA10。\n'
            '    _dep = (df["sum_bidv"] + df["sum_askv"]) / df["sum_amt"].where(df["sum_amt"] > 0)\n'
            '    df["c_depth"] = -np.log1p(_dep.where(_dep > 0)).groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "437_code")
    s = add_keep(s, ["c_depth"], "437")
    return add_sig(s, "437", "c_depth", 0.15)


def c438(s):
    """盘口不平衡的窗口内波动。"""
    s = rich(s, "438")
    code = ('    # 盘口不平衡的窗口内波动（挂单结构不稳定性），EWMA10。\n'
            '    _m = df["sum_ob"] / 30.0\n'
            '    _v = (df["sum_ob2"] / 30.0 - _m * _m).clip(lower=0)\n'
            '    df["c_obvol"] = np.sqrt(_v).groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "438_code")
    s = add_keep(s, ["c_obvol"], "438")
    return add_sig(s, "438", "c_obvol", 0.15)


def c439(s):
    """量价弹性（Kyle lambda 代理）。"""
    s = rich(s, "439")
    code = ('    # 量价弹性 Kyle lambda 代理：窗口内 cov(r, v)/var(v)，40 窗口均值取负。\n'
            '    _n = 30.0\n'
            '    _mv = df["sum_vol"] / _n\n'
            '    _mr = df["sum_r1"] / _n\n'
            '    _cov = df["sum_rv"] / _n - _mr * _mv\n'
            '    _var = (df["sum_v2"] / _n - _mv * _mv).clip(lower=0)\n'
            '    _lam = _cov / _var.where(_var > 0)\n'
            '    df["c_lambda"] = -roll(_lam.where(np.isfinite(_lam)), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "439_code")
    s = add_keep(s, ["c_lambda"], "439")
    return add_sig(s, "439", "c_lambda", 0.15)


def c440(s):
    """分钟收益已实现峰度。"""
    s = rich(s, "440")
    code = ('    # 分钟收益已实现峰度：Σr⁴/Σr²²（40 窗口均值取负=偏好无厚尾）。\n'
            '    _k = df["sum_r1q"] / (df["sum_r1sq"] ** 2).where(df["sum_r1sq"] > 0)\n'
            '    df["c_kurt"] = -roll(_k.where(np.isfinite(_k)), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "440_code")
    s = add_keep(s, ["c_kurt"], "440")
    return add_sig(s, "440", "c_kurt", 0.15)


def resid(s, num, col, w):
    s = add_keep(s, [col + "_p"], num + "_pk")
    r = ('    result["%s_p"] = result["%s"] - peer_mean("%s").fillna(result.groupby("date")["%s"].transform("mean"))\n'
         % (col, col, col, col))
    s = repl(s, A_LADDER, A_LADDER + r, num + "_resid")
    return add_sig(s, num, col + "_p", w)


def c442(s):
    """上影线占比的同伴残差。"""
    return resid(s, "442", "c_upsh", 0.15)


def c443(s):
    """r_rv16 的同伴残差。"""
    return resid(s, "443", "r_rv16", 0.20)


def c441(s):
    """同伴滞后两格的收益溢出。"""
    code = ('    # 同伴滞后两格（约 1 小时）的收益溢出。\n'
            '    _dd = result["date"].dt.normalize()\n'
            '    _pr = peer_mean("r_30").fillna(result.groupby("date")["r_30"].transform("mean"))\n'
            '    df2 = pd.DataFrame({"instrument": result["instrument"], "day": _dd, "pv": _pr})\n'
            '    df2 = df2.sort_values(["instrument", "day"])\n'
            '    _g2 = df2.groupby("instrument", sort=False)["pv"]\n'
            '    result["c_spill2"] = (_g2.shift(1) + _g2.shift(2)).reindex(result.index) / 2.0\n')
    s = repl(s, A_LADDER, A_LADDER + code, "441_code")
    s = add_keep(s, ["c_spill2"], "441")
    return add_sig(s, "441", "c_spill2", 0.10)


CANDS = [("437", "报价深度成交比", c437),
         ("438", "盘口不平衡波动", c438),
         ("439", "量价弹性", c439),
         ("440", "分钟收益峰度", c440),
         ("441", "同伴滞后两格溢出", c441),
         ("442", "同伴残差上影线", c442),
         ("443", "同伴残差二日反转", c443)]

if __name__ == "__main__":
    for num, name, fn in CANDS:
        s = fn(B373.read_text(encoding="utf-8"))
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)
