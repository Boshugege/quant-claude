# -*- coding: utf-8 -*-
"""第四十七批（313+）：最后的未试维度——相对买卖价差（水平/趋势）、单笔成交额离散度、日内成交额时间重心。

骨架 = 293。313/314/315/316 为单杠杆，317 为 313 的反号对照，318 为 313+315 叠加。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "293_规模加权同伴幂二EWMA十上影线加平均单笔二日尺度与实体占比.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n'
A_CBODY = '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
A_KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'


def add_spread(s, num):
    snap = ('            CASE WHEN bp1 > 0 AND ap1 > bp1 THEN (ap1 - bp1) / ((ap1 + bp1) / 2.0) ELSE NULL END AS spread_c,\n')
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, '            AVG(spread_c) AS avg_spread,\n' + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.rstrip("\n") + ", f.avg_spread\n", num + "_sel")
    s = repl(s, A_CAST, '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals", "avg_spread"]:\n', num + "_cast")
    return s


def finish(s, code, cols, weights, num):
    s = repl(s, A_CBODY, A_CBODY + code, num + "_comp")
    s = repl(s, A_KEEP, A_KEEP.rstrip("\n").replace('"c_body"]', '"c_body", %s]' % ", ".join('"%s"' % c for c in cols)) + "\n", num + "_keep")
    base_cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"] + list(cols)
    base_w = [1.0, 0.30, 0.15, 0.20, 0.15] + list(weights)
    s = repl(s, A_SIG, '    result["signal"] = rank_blend([%s], [%s])\n' % (
        ", ".join('"%s"' % c for c in base_cols), ", ".join("%.2f" % w for w in base_w)), num + "_sig")
    return s


CODE313 = ('    # 相对买卖价差水平：(卖一−买一)/中间价 的 40 窗口均值（正号，流动性差＝反转更强）。\n'
           '    df["c_spread"] = roll(df["avg_spread"], 40, "mean", 10)\n')
CODE314 = ('    # 相对价差趋势：近 10 窗口均值 − 近 40 窗口均值（取负）。\n'
           '    df["c_spreadtr"] = -(roll(df["avg_spread"], 10, "mean", 5) - roll(df["avg_spread"], 40, "mean", 10))\n')
CODE315 = ('    # 单笔成交额离散度：分钟平均单笔成交额的 40 窗口标准差（取负）。\n'
           '    _at = df["sum_amt"] / df["sum_deals"].where(df["sum_deals"] > 0)\n'
           '    df["c_atrsd"] = -roll(_at, 40, "std", 10)\n')
CODE316 = ('    # 日内成交额时间重心：Σ(段序×段成交额)/Σ段成交额（日级），取负。\n'
           '    _segnum = df["date"].dt.strftime("%H%M").map({"1000": 2, "1030": 3, "1100": 4, "1130": 5, "1330": 6, "1400": 7, "1430": 8})\n'
           '    _num = (df["sum_amt"] * _segnum.fillna(1)).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
           '    _den = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
           '    df["c_vtc"] = -(_num / _den.where(_den > 0))\n')


def c313(s):
    s = add_spread(s, "313"); return finish(s, CODE313, ["c_spread"], [0.15], "313")
def c314(s):
    s = add_spread(s, "314"); return finish(s, CODE314, ["c_spreadtr"], [0.15], "314")
def c315(s):
    return finish(s, CODE315, ["c_atrsd"], [0.10], "315")
def c316(s):
    return finish(s, CODE316, ["c_vtc"], [0.15], "316")
def c317(s):
    s = add_spread(s, "317")
    return finish(s, "    df[\"c_spreadn\"] = -roll(df[\"avg_spread\"], 40, \"mean\", 10)\n", ["c_spreadn"], [0.15], "317")
def c318(s):
    s = add_spread(s, "318")
    return finish(s, '    df["c_spread"] = roll(df["avg_spread"], 40, "mean", 10)\n' + CODE315, ["c_spread", "c_atrsd"], [0.15, 0.10], "318")


CANDS = [("313", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与相对价差", c313),
         ("314", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与价差趋势", c314),
         ("315", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与单笔额离散度", c315),
         ("316", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与成交额时间重心", c316),
         ("317", "相对价差反号", c317),
         ("318", "相对价差加单笔额离散度", c318)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
