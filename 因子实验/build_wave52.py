# -*- coding: utf-8 -*-
"""第五十二批（360+）：在 353（当前合法最好，258 + r_rv16 + c_body）上做纯 pandas 变体。

**本批所有文件与 353 的 SQL 逐字相同**，只有 pandas 部分不同 → 评测机 2.4.2 的 SQL 缓存只需执行一次。
方向：换手率代理作信号 / 上影线 EWMA 形态 / 两日尺度权重 / 委托笔数。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "353_同伴上影线平均单笔二日尺度与实体占比.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
    return t.replace(old, new, 1)


A_UPSH = '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
CODE_RVOL = ('    # 换手率代理：近 40 窗口成交量 / 自己过去 240 日的 40 窗口平均成交量（相对自身历史，无需股本）。\n'
             '    _v40 = roll(df["sum_vol"], 40, "sum", 10)\n'
             '    df["c_rvol"] = _v40 / roll(_v40, 1920, "mean", 240).where(lambda v: v > 0)\n')
SIG353 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
NEU15 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'


def add_keep(s, cols, num):
    import re
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
    assert m, num
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


def add_rvol(s, num):
    s = repl(s, A_UPSH, A_UPSH + CODE_RVOL, num + "_rvol")
    return add_keep(s, ["c_rvol"], num)


def c360(s):
    """换手率代理作为**信号分量**（正权重 0.15）——354 显示它作中性化特征有害，说明信号里含正向的换手信息。"""
    s = add_rvol(s, "360")
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_rvol"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n'
    return repl(s, SIG353, sig, "360_sig")


def c361(s):
    """换手率代理作信号分量，权重 0.30。"""
    s = add_rvol(s, "361")
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_rvol"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.30])\n'
    return repl(s, SIG353, sig, "361_sig")


def c362(s):
    """上影线改用 293 的 EWMA(span=10) 形态（258 用的是 40 窗口均值）。"""
    new = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'
    return repl(s, A_UPSH, new, "362_upsh")


def c363(s):
    """换手率代理进中性化特征，但强度减半（1.5 → 0.75），检验 354 的伤害是否剂量依赖。"""
    s = add_rvol(s, "363")
    s = repl(s, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]\n',
             '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_rvol"]\n', "363_chars")
    return repl(s, NEU15, '    result["factor"] = neutralize("signal", CHARS4, 0.75)\n', "363_frac")


def c364(s):
    """两日尺度 r_rv16 权重 0.20 → 0.30（293 时代该权重是 0.20，此处重扫）。"""
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.30, 0.15])\n'
    return repl(s, SIG353, sig, "364_sig")


def c365(s):
    """委托笔数（log，40 窗口均值）作为信号分量 0.15——258 算了 c_deals 但从未使用。"""
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_deals"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n'
    return repl(s, SIG353, sig, "365_sig")


CANDS = [("360", "换手率代理作信号015", c360),
         ("361", "换手率代理作信号030", c361),
         ("362", "上影线EWMA十", c362),
         ("363", "换手率代理中性化075", c363),
         ("364", "两日尺度权重030", c364),
         ("365", "委托笔数作信号015", c365)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
