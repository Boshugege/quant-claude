# -*- coding: utf-8 -*-
"""第四十八批（319+）：1 分钟粒度（子截面内）的新维度——跳跃性、成交额集中度、微观已实现波动。

之前所有分量都在 30 分钟段粒度上聚合；本批第一次用**分钟内**的量：
- 分钟跳跃性 = max|分钟收开| / mean|分钟收开|（窗口内），取负
- 分钟成交额 HHI = Σamt²/(Σamt)²（窗口内），取负
- 分钟最大额占比 = max amt / Σamt，取负
- 微观已实现波动 = Σ(分钟内收开收益²)，取负

319 是**纯新轴**（只用跳跃性，不带反转骨架），用来测它与 104 的截面相关能否低于 0.8；
320-324 是在 293 骨架上加这些新维度。
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

WIN_COLS = ["avg_absr", "max_absr", "sum_r1sq", "sum_amt2", "max_amt"]
SEL_EXTRA = ", f.avg_absr, f.max_absr, f.sum_r1sq, f.sum_amt2, f.max_amt"
CAST_NEW = ('    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals", '
            '"avg_absr", "max_absr", "sum_r1sq", "sum_amt2", "max_amt"]:\n')


def add_minute_sql(s, num):
    snap = ('            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr1_c,\n'
            '            CASE WHEN px_open > 0 AND px_close > 0 THEN ((px_close - px_open) / px_open) * ((px_close - px_open) / px_open) ELSE 0 END AS sqr1_c,\n'
            '            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN amt * amt ELSE 0 END AS amt2_c,\n')
    win = ('            AVG(absr1_c) AS avg_absr,\n'
           '            MAX(absr1_c) AS max_absr,\n'
           '            SUM(sqr1_c) AS sum_r1sq,\n'
           '            SUM(amt2_c) AS sum_amt2,\n'
           '            MAX(amt_c) AS max_amt,\n')
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.rstrip("\n") + SEL_EXTRA + "\n", num + "_sel")
    s = repl(s, A_CAST, CAST_NEW, num + "_cast")
    return s


CODE_JUMP = ('    # 分钟跳跃性：窗口内 max|分钟收开| / mean|分钟收开|，近 40 窗口均值取负。\n'
             '    _jmp = df["max_absr"] / df["avg_absr"].where(df["avg_absr"] > 0)\n'
             '    _jmp = _jmp.where(lambda v: (v > 0) & np.isfinite(v))\n'
             '    df["c_jump"] = -roll(_jmp, 40, "mean", 10)\n')
CODE_HHI = ('    # 分钟成交额 HHI：窗口内 Σamt²/(Σamt)²，近 40 窗口均值取负。\n'
            '    _hhi = df["sum_amt2"] / (df["sum_amt"] ** 2).where(df["sum_amt"] > 0)\n'
            '    df["c_amthhi"] = -roll(_hhi.where(lambda v: v > 0), 40, "mean", 10)\n')
CODE_MAXS = ('    # 分钟最大额占比：窗口内 max amt / Σamt，近 40 窗口均值取负。\n'
             '    _mx = df["max_amt"] / df["sum_amt"].where(df["sum_amt"] > 0)\n'
             '    df["c_maxshare"] = -roll(_mx.where(lambda v: v > 0), 40, "mean", 10)\n')
CODE_MRV = ('    # 微观已实现波动：窗口内 Σ(分钟内收开收益²)，近 40 窗口均值取负。\n'
            '    df["c_micrv"] = -roll(df["sum_r1sq"], 40, "mean", 10)\n')


def finish(s, codes, cols, weights, num, standalone=False):
    for i, c in enumerate(codes):
        s = repl(s, A_CBODY, A_CBODY + c, "%s_comp%d" % (num, i)) if i == 0 else repl(s, A_CBODY, A_CBODY + c, "%s_comp%d" % (num, i))
    s = repl(s, A_KEEP, A_KEEP.rstrip("\n").replace('"c_body"]', '"c_body", %s]' % ", ".join('"%s"' % c for c in cols)) + "\n", num + "_keep")
    if standalone:
        sig = '    result["signal"] = rank_blend([%s], [%s])\n' % (
            ", ".join('"%s"' % c for c in cols), ", ".join("%.2f" % w for w in weights))
    else:
        base_cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"] + list(cols)
        base_w = [1.0, 0.30, 0.15, 0.20, 0.15] + list(weights)
        sig = '    result["signal"] = rank_blend([%s], [%s])\n' % (
            ", ".join('"%s"' % c for c in base_cols), ", ".join("%.2f" % w for w in base_w))
    s = repl(s, A_SIG, sig, num + "_sig")
    return s


def c319(s):
    s = add_minute_sql(s, "319")
    return finish(s, [CODE_JUMP], ["c_jump"], [1.0], "319", standalone=True)
def c320(s):
    s = add_minute_sql(s, "320")
    return finish(s, [CODE_JUMP], ["c_jump"], [0.15], "320")
def c321(s):
    s = add_minute_sql(s, "321")
    return finish(s, [CODE_HHI], ["c_amthhi"], [0.15], "321")
def c322(s):
    s = add_minute_sql(s, "322")
    return finish(s, [CODE_MAXS], ["c_maxshare"], [0.15], "322")
def c323(s):
    s = add_minute_sql(s, "323")
    return finish(s, [CODE_JUMP, CODE_HHI], ["c_jump", "c_amthhi"], [0.15, 0.15], "323")
def c324(s):
    s = add_minute_sql(s, "324")
    return finish(s, [CODE_MRV], ["c_micrv"], [0.10], "324")


CANDS = [("319", "分钟跳跃性", c319),
         ("320", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与分钟跳跃性", c320),
         ("321", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与分钟成交额HHI", c321),
         ("322", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与分钟最大额占比", c322),
         ("323", "分钟跳跃性加成交额HHI", c323),
         ("324", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与微观已实现波动", c324)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
