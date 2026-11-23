# -*- coding: utf-8 -*-
"""第五十四批（375+）：新方向（不为缓存复用思路）。

375/376：第四十八批的分钟成交额集中度（322/321）在被作废的家族里与头部打平，本批用合法口径（353 骨架）重测。
377：分钟实体方向性（Σ(收-开)/Σ(高-低)）——主动买盘/卖盘方向的分钟代理（不是 |实体|，是有符号）。
378：上下影线不对称（低位影线占比）——上影线已用，下影线在合法家族里从未单独测过。
379：上午/下午波动比——日内波动的时间分布（形状），非水平。
380：相对成交额（窗口成交额 ÷ 自身过去 240 日均额）——换手率代理的成交额版本（第五十一批用的是成交量版本）。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "353_同伴上影线平均单笔二日尺度与实体占比.py"


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


SIG353 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
A_UPSH = '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'


def add_sql_cols(s, num, snap_extra="", win_extra="", sel_names=(), cast_names=()):
    if snap_extra:
        s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap_extra, num + "_snap")
    if win_extra:
        s = repl(s, A_DEALS_WIN, win_extra + A_DEALS_WIN, num + "_win")
    if sel_names:
        s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, " + ", ".join("f.%s" % c for c in sel_names) + "\n"), num + "_sel")
    if cast_names:
        s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", ' + ", ".join('"%s"' % c for c in cast_names) + "]"), num + "_cast")
    return s


def add_sig(s, num, col, w):
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "%s"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % (col, w))
    return repl(s, SIG353, sig, num + "_sig")


def c375(s):
    """分钟最大额占比（322 的杠杆）。"""
    s = add_sql_cols(s, "375",
                     snap_extra='            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN amt ELSE 0 END AS amt_c,\n',
                     win_extra='            MAX(amt_c) AS max_amt,\n',
                     sel_names=("max_amt",), cast_names=("max_amt",))
    code = ('    # 分钟最大额占比：窗口内 max amt / Σamt，40 窗口均值取负。\n'
            '    _mx = df["max_amt"] / df["sum_amt"].where(df["sum_amt"] > 0)\n'
            '    df["c_maxshare"] = -roll(_mx.where(lambda v: v > 0), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "375_code")
    s = add_keep(s, ["c_maxshare"], "375")
    return add_sig(s, "375", "c_maxshare", 0.15)


def c376(s):
    """分钟成交额 HHI（321 的杠杆）。"""
    s = add_sql_cols(s, "376",
                     snap_extra='            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN amt * amt ELSE 0 END AS amt2_c,\n',
                     win_extra='            SUM(amt2_c) AS sum_amt2,\n',
                     sel_names=("sum_amt2",), cast_names=("sum_amt2",))
    code = ('    # 分钟成交额 HHI：窗口内 Σamt²/(Σamt)²，40 窗口均值取负。\n'
            '    _hhi = df["sum_amt2"] / (df["sum_amt"] ** 2).where(df["sum_amt"] > 0)\n'
            '    df["c_amthhi"] = -roll(_hhi.where(lambda v: v > 0), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "376_code")
    s = add_keep(s, ["c_amthhi"], "376")
    return add_sig(s, "376", "c_amthhi", 0.15)


def c377(s):
    """分钟实体方向性（有符号）：Σ(收-开)/Σ(高-低)。"""
    s = add_sql_cols(s, "377",
                     snap_extra='            CASE WHEN px_open > 0 AND px_close > 0 THEN px_close - px_open ELSE 0 END AS sbody_c,\n',
                     win_extra='            SUM(sbody_c) AS sum_sbody,\n',
                     sel_names=("sum_sbody",), cast_names=("sum_sbody",))
    code = ('    # 分钟实体方向性：窗口内 Σ(收-开)/Σ(高-低)，40 窗口均值（有符号，主动方向代理）。\n'
            '    df["c_bodydir"] = roll(df["sum_sbody"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "377_code")
    s = add_keep(s, ["c_bodydir"], "377")
    return add_sig(s, "377", "c_bodydir", 0.15)


def c378(s):
    """下影线占比（低位承接）：窗口内 Σ(实体下沿-最低)/Σ(高-低)，40 窗口均值取负。"""
    code = ('    # 下影线占比：窗口内 Σ(min(开,收)-最低)/Σ(高-低)，40 窗口均值取负。\n'
            '    df["c_dnsh"] = -roll(df["sum_dnsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "378_code")
    s = add_keep(s, ["c_dnsh"], "378")
    return add_sig(s, "378", "c_dnsh", 0.15)


def c379(s):
    """上午/下午波动比（日内波动的形状）。"""
    code = ('    # 上午/下午波动比：当日 13:30–14:30 振幅 ÷ 09:30–11:30 振幅，40 窗口均值。\n'
            '    df["hhmm"] = df["date"].dt.strftime("%H%M").astype(int)\n'
            '    df["_am"] = df["sum_rng"].where(df["hhmm"] <= 1130, 0.0)\n'
            '    df["_pm"] = df["sum_rng"].where(df["hhmm"] >= 1330, 0.0)\n'
            '    _dayam = df.groupby(["instrument", "day"], sort=False)["_am"].transform("sum")\n'
            '    _daypm = df.groupby(["instrument", "day"], sort=False)["_pm"].transform("sum")\n'
            '    _ratio = np.log((_daypm + 1e-12) / (_dayam + 1e-12))\n'
            '    df["c_vshape"] = roll(_ratio, 40, "mean", 10)\n')
    s = repl(s, A_RV8, A_RV8 + code, "379_code")
    s = add_keep(s, ["c_vshape"], "379")
    return add_sig(s, "379", "c_vshape", 0.15)


def c380(s):
    """相对成交额（成交额版本）：窗口成交额 ÷ 自身过去 240 日的 40 窗口平均成交额。"""
    code = ('    # 相对成交额：近 40 窗口成交额 / 自己过去 240 日的 40 窗口平均成交额。\n'
            '    _a40 = roll(df["sum_amt"], 40, "sum", 10)\n'
            '    df["c_ramt"] = _a40 / roll(_a40, 1920, "mean", 240).where(lambda v: v > 0)\n')
    s = repl(s, A_UPSH, A_UPSH + code, "380_code")
    s = add_keep(s, ["c_ramt"], "380")
    return add_sig(s, "380", "c_ramt", 0.15)


CANDS = [("375", "加分钟最大额占比", c375),
         ("376", "加分钟成交额HHI", c376),
         ("377", "加分钟实体方向性", c377),
         ("378", "加下影线占比", c378),
         ("379", "加上下午波动比", c379),
         ("380", "加相对成交额", c380)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
