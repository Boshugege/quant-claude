# -*- coding: utf-8 -*-
"""第六十批（421+）：换新轴，全部是此前合法骨架里没碰过的信息源（底座 373）。

- 421/422 五档挂单量不平衡（正/反两向）：盘口深度是纯订单簿信息，与成交量类信号不同源。
- 423 五档挂单量不平衡的同伴残差：把股票特有的盘口压制提出来（跳跃轴的同伴残差曾从稀释变等价）。
- 424 一档（L1）挂单不平衡：近档对短期价格压力更敏感。
- 425 分钟收益已实现偏度：窗口内 Σr³/Σr²^1.5，刻画日内冲高/跳水的形态。
- 426 同伴上一时段收益溢出（lead-lag）：同伴延迟一格的收益预测自身下一格（HKS 同时段延续的日内版）。
- 427 单笔成交额离散度：窗口内每分钟平均单笔成交额的变异系数（大单/小单结构的不稳定性）。
- 428 分钟收益已实现偏度的同伴残差。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
B373 = D / "362_上影线EWMA十.py"


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
A_RAW = '            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'


def transparent(s, num, snap, win, names):
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in names) + "\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in names) + "]"), num + "_cast")
    return s


def add_sig(s, num, cols, weights=None):
    if isinstance(cols, str):
        cols, weights = [cols], [None]
    if not weights or any(w is None for w in weights):
        weights = [0.15] * len(cols)
    parts = ", ".join('"%s"' % c for c in cols)
    wts = ", ".join("%.2f" % w for w in weights)
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", %s], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, %s])\n' % (parts, wts))
    return repl(s, SIG, sig, num + "_sig")


# ---------- 盘口深度（订单簿） ----------
OB_RAW = ('            CAST(COALESCE(bid_volume1, 0) AS DOUBLE) AS bv1,\n'
          '            CAST(COALESCE(bid_volume2, 0) AS DOUBLE) AS bv2,\n'
          '            CAST(COALESCE(bid_volume3, 0) AS DOUBLE) AS bv3,\n'
          '            CAST(COALESCE(bid_volume4, 0) AS DOUBLE) AS bv4,\n'
          '            CAST(COALESCE(bid_volume5, 0) AS DOUBLE) AS bv5,\n'
          '            CAST(COALESCE(ask_volume1, 0) AS DOUBLE) AS av1,\n'
          '            CAST(COALESCE(ask_volume2, 0) AS DOUBLE) AS av2,\n'
          '            CAST(COALESCE(ask_volume3, 0) AS DOUBLE) AS av3,\n'
          '            CAST(COALESCE(ask_volume4, 0) AS DOUBLE) AS av4,\n'
          '            CAST(COALESCE(ask_volume5, 0) AS DOUBLE) AS av5,\n')
OB_SNAP = ('            (bv1 + bv2 + bv3 + bv4 + bv5) AS bidv_c,\n'
           '            (av1 + av2 + av3 + av4 + av5) AS askv_c,\n'
           '            bv1 AS bidv1_c,\n'
           '            av1 AS askv1_c,\n')
OB_WIN = ('            SUM(bidv_c) AS sum_bidv,\n            SUM(askv_c) AS sum_askv,\n'
          '            SUM(bidv1_c) AS sum_bidv1,\n            SUM(askv1_c) AS sum_askv1,\n')


def ob_base(s, num):
    s = repl(s, A_RAW, A_RAW[:-1] + ",\n" + OB_RAW, num + "_raw")
    return transparent(s, num, OB_SNAP, OB_WIN, ("sum_bidv", "sum_askv", "sum_bidv1", "sum_askv1"))


def c421(s):
    s = ob_base(s, "421")
    code = ('    # 五档挂单量不平衡的窗口均值（EWMA10）。买方挂单厚 → 瞬时买压 → 反转口径取负。\n'
            '    _den = (df["sum_bidv"] + df["sum_askv"])\n'
            '    _ob = (df["sum_bidv"] - df["sum_askv"]) / _den.where(_den > 0)\n'
            '    df["c_ob"] = -_ob.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "421_code")
    s = add_keep(s, ["c_ob"], "421")
    return add_sig(s, "421", "c_ob")


def c422(s):
    s = c421(s)
    return s.replace('    df["c_ob"] = -_ob', '    df["c_ob"] = _ob', 1)


def c423(s):
    s = c421(s)
    s = add_keep(s, ["c_ob_p"], "423_pk")
    resid = '    result["c_ob_p"] = result["c_ob"] - peer_mean("c_ob").fillna(result.groupby("date")["c_ob"].transform("mean"))\n'
    s = repl(s, A_LADDER, A_LADDER + resid, "423_resid")
    return s.replace('"c_ob"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15]', '"c_ob_p"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15]', 1)


def c424(s):
    s = ob_base(s, "424")
    code = ('    # 一档挂单量不平衡（EWMA10），反转口径取负。\n'
            '    _den = (df["sum_bidv1"] + df["sum_askv1"])\n'
            '    _ob1 = (df["sum_bidv1"] - df["sum_askv1"]) / _den.where(_den > 0)\n'
            '    df["c_ob1"] = -_ob1.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "424_code")
    s = add_keep(s, ["c_ob1"], "424")
    return add_sig(s, "424", "c_ob1")


# ---------- 分钟收益已实现偏度 ----------
SK_SNAP = ('            CASE WHEN px_open > 0 THEN (px_close - px_open) / px_open ELSE 0 END AS r1_c,\n'
           '            CASE WHEN px_open > 0 THEN ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) ELSE 0 END AS r1sq_c,\n'
           '            CASE WHEN px_open > 0 THEN ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) ELSE 0 END AS r1cb_c,\n')
SK_WIN = '            SUM(r1sq_c) AS sum_r1sq,\n            SUM(r1cb_c) AS sum_r1cb,\n'


def sk_base(s, num):
    return transparent(s, num, SK_SNAP, SK_WIN, ("sum_r1sq", "sum_r1cb"))


def c425(s):
    s = sk_base(s, "425")
    code = ('    # 分钟收益已实现偏度：Σr³ / (Σr²)^1.5（对正偏（冲高）取负，反转口径）。\n'
            '    _sk = df["sum_r1cb"] / (df["sum_r1sq"] ** 1.5).where(df["sum_r1sq"] > 0)\n'
            '    _sk = _sk.where(np.isfinite(_sk))\n'
            '    df["c_skew"] = -_sk.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    s = repl(s, A_UPSH, A_UPSH + code, "425_code")
    s = add_keep(s, ["c_skew"], "425")
    return add_sig(s, "425", "c_skew")


def c428(s):
    s = c425(s)
    s = add_keep(s, ["c_skew_p"], "428_pk")
    resid = '    result["c_skew_p"] = result["c_skew"] - peer_mean("c_skew").fillna(result.groupby("date")["c_skew"].transform("mean"))\n'
    s = repl(s, A_LADDER, A_LADDER + resid, "428_resid")
    return s.replace('"c_skew"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15]', '"c_skew_p"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15]', 1)


# ---------- 单笔成交额离散度 ----------
AT_SNAP = ('            CASE WHEN deals > 0 AND vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN amt / deals ELSE 0 END AS at_c,\n'
           '            CASE WHEN deals > 0 AND vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN (amt / deals) * (amt / deals) ELSE 0 END AS atsq_c,\n'
           '            CASE WHEN deals > 0 AND vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN 1 ELSE 0 END AS atn_c,\n')
AT_WIN = ('            SUM(at_c) AS sum_at,\n            SUM(atsq_c) AS sum_atsq,\n'
          '            SUM(atn_c) AS cnt_at,\n')


def c427(s):
    s = transparent(s, "427", AT_SNAP, AT_WIN, ("sum_at", "sum_atsq", "cnt_at"))
    code = ('    # 单笔成交额离散度：窗口内每分钟平均单笔成交额的变异系数（40 窗口均值，取负=偏好稳定）。\n'
            '    _n = df["cnt_at"].where(df["cnt_at"] > 1)\n'
            '    _m = df["sum_at"] / _n\n'
            '    _v = (df["sum_atsq"] / _n - _m * _m).clip(lower=0)\n'
            '    _cv = np.sqrt(_v) / _m.where(_m > 0)\n'
            '    df["c_atdisp"] = -roll(_cv.where(np.isfinite(_cv)), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "427_code")
    s = add_keep(s, ["c_atdisp"], "427")
    return add_sig(s, "427", "c_atdisp")


# ---------- 同伴上一时段收益溢出 ----------
def c426(s):
    code = ('    # 同伴上一时段收益溢出：同伴在上一格的平均收益（lead-lag，HKS 同时段延续的日内版）。\n'
            '    _dd = result["date"].dt.normalize()\n'
            '    _pr = peer_mean("r_30").fillna(result.groupby("date")["r_30"].transform("mean"))\n'
            '    df2 = pd.DataFrame({"instrument": result["instrument"], "day": _dd, "pv": _pr})\n'
            '    df2 = df2.sort_values(["instrument", "day"])\n'
            '    result["c_spill"] = df2.groupby("instrument", sort=False)["pv"].shift(1).reindex(result.index)\n')
    s = repl(s, A_LADDER, A_LADDER + code, "426_code")
    s = add_keep(s, ["c_spill"], "426")
    return add_sig(s, "426", ["c_spill"], [0.10])


CANDS = [("421", "五档挂单不平衡", c421),
         ("422", "五档挂单不平衡反向", c422),
         ("423", "同伴残差五档挂单不平衡", c423),
         ("424", "一档挂单不平衡", c424),
         ("425", "分钟收益已实现偏度", c425),
         ("426", "同伴上一时段收益溢出", c426),
         ("427", "单笔成交额离散度", c427),
         ("428", "同伴残差分钟收益偏度", c428)]

if __name__ == "__main__":
    for num, name, fn in CANDS:
        s = fn(B373.read_text(encoding="utf-8"))
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)
