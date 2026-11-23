# -*- coding: utf-8 -*-
"""第五十一批（353+）：不读暴露表，用**行情代理**找回此前依赖流通市值的杠杆。

模板 = 258（当前唯一合法头部：长窗口软同伴 + 上影线 0.30 + 平均单笔 0.15 + 四特征中性化）。
本批把 293 相对 258 多出来的东西用纯行情代理复刻：
- r_rv16（约 2 日尺度 VWAP 反转）、c_body（实体占比）——只用 bar1m。
- 换手率代理 c_rvol = 近 40 窗口成交量 / 自己过去 240 日的平均 40 窗口成交量（相对自身历史，不需要股本）。
- 市值二次项代理 c_sz2 = ln(近 40 窗口成交额)²（官方 SIZE 只覆盖线性部分）。
- 规模加权同伴：同伴权重改用近 60 日日均成交额。

新方向（与 104 的截面相关预期 <0.8）：357/358 的成交额加权同伴。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "258_长上影线加平均单笔成交额.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
    return t.replace(old, new, 1)


A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
A_UPSH = '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
A_KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals"]\n'
A_SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr"], [1.0, 0.30, 0.15])\n'
A_CHARS = '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]\n'
A_SIG_EXT = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
A_WTS = '        wts = np.clip(cvals, 0.0, None) ** PEER_POWER   # 软同伴：正相关度加权\n'

CODE_RV16 = '    df["r_rv16"] = rolling_vwap_rev(16)\n'
CODE_BODY = ('    # 实体占比（分钟 |收-开| 之和 / 振幅之和），40 窗口均值取负。\n'
             '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')
CODE_RVOL = ('    # 换手率代理：近 40 窗口成交量 / 自己过去 240 日的 40 窗口平均成交量（相对自身历史，无需股本）。\n'
             '    _v40 = roll(df["sum_vol"], 40, "sum", 10)\n'
             '    df["c_rvol"] = _v40 / roll(_v40, 1920, "mean", 240).where(lambda v: v > 0)\n')
CODE_SZ2 = ('    # 市值二次项代理：ln(近 40 窗口成交额) 的平方（官方 SIZE 只覆盖线性部分）。\n'
            '    df["c_sz2"] = np.log1p(roll(df["sum_amt"], 40, "sum", 10)) ** 2\n')


def add_keep(s, cols, num):
    """幂等：把 cols 追加到当前 keep 行（若已存在则跳过）。"""
    m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
    assert m, num + "_keep_anchor"
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


def extend_signal(s, num):
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
    return repl(s, A_SIG, sig, num + "_sig")


def core(s, num):
    """加 r_rv16 + c_body（信号扩展，纯行情）。"""
    s = repl(s, A_RV8, A_RV8 + CODE_RV16, num + "_rv16")
    s = repl(s, A_UPSH, A_UPSH + CODE_BODY, num + "_body")
    s = add_keep(s, ["r_rv16", "c_body"], num)
    return extend_signal(s, num)


def set_chars(s, extras, num):
    cols = ["c_size", "c_ivol", "c_illiq", "c_lowpx"] + list(extras)
    return repl(s, A_CHARS, '    CHARS4 = [%s]\n' % ", ".join('"%s"' % c for c in cols), num + "_chars")


def add_rvol(s, num):
    s = repl(s, A_UPSH, A_UPSH + CODE_RVOL, num + "_rvolcode")
    return add_keep(s, ["c_rvol"], num)


def add_sz2(s, num):
    s = repl(s, A_UPSH, A_UPSH + CODE_SZ2, num + "_sz2code")
    return add_keep(s, ["c_sz2"], num)


def amount_weighted_peers(s, num):
    """把同伴权重从 corr^power 改成近 60 日日均成交额（选择仍按相关性）。"""
    prep = ('    # 近 60 日日均成交额（用于规模加权同伴，同月内取月初前 60 个交易日）。\n'
            '    _damt = df.groupby(["instrument", "day"], sort=False)["sum_amt"].sum().unstack("instrument")\n'
            '    amt60_by_month = {}\n'
            '    for _m in out_months:\n'
            '        _in = all_days[day_month == _m]\n'
            '        if len(_in) == 0:\n'
            '            continue\n'
            '        _prev = _damt.loc[_damt.index < _in[0]].tail(60)\n'
            '        amt60_by_month[_m] = _prev.mean(axis=0)\n')
    s = repl(s, "    for m in out_months:\n", prep + "    for m in out_months:\n", num + "_prep")
    neww = ('        _a60 = amt60_by_month.get(m)\n'
            '        _awf = np.clip(_a60.reindex(peer_insts).to_numpy(np.float64), 0.0, None) if _a60 is not None else np.ones(len(peer_insts))\n'
            '        _aw = np.nan_to_num(_awf, nan=0.0)[sub]   # 只取通过相关性门槛的子集\n'
                        '        wts = np.where(cvals > 0, _aw[:, None] * np.clip(cvals, 0.0, None), 0.0)\n')
    return repl(s, A_WTS, neww, num + "_wts")


def c353(s):
    return core(s, "353")
def c354(s):
    s = core(s, "354"); s = add_rvol(s, "354"); return set_chars(s, ["c_rvol"], "354")
def c355(s):
    s = core(s, "355"); s = add_sz2(s, "355"); return set_chars(s, ["c_sz2"], "355")
def c356(s):
    s = core(s, "356"); s = add_rvol(s, "356"); s = add_sz2(s, "356"); return set_chars(s, ["c_rvol", "c_sz2"], "356")
def c357(s):
    s = core(s, "357"); return amount_weighted_peers(s, "357")
def c358(s):
    s = core(s, "358"); s = add_rvol(s, "358"); s = add_sz2(s, "358"); s = set_chars(s, ["c_rvol", "c_sz2"], "358")
    return amount_weighted_peers(s, "358")

def c359(s):
    """放量加权反转（待做清单 ②→85 号）：反转阶梯 × 相对成交量，作为新的交互分量。"""
    s = core(s, "359")
    s = add_rvol(s, "359")
    A_LAD = '    result["ladder"] = peer_ladder()\n'
    inter = ('    _lr = result.groupby("date")["ladder"].rank(pct=True) - 0.5\n'
             '    _vr = result.groupby("date")["c_rvol"].rank(pct=True) - 0.5\n'
             '    result["rev_x_vol"] = _lr * _vr\n')
    s = repl(s, A_LAD, A_LAD + inter, "359_inter")
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "rev_x_vol"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.20])\n'
    s = s.replace(A_SIG_EXT, sig, 1)
    return s

CANDS = [("353", "同伴上影线平均单笔二日尺度与实体占比", c353),
         ("354", "加换手率代理", c354),
         ("355", "加市值二次项代理", c355),
         ("356", "加换手率与市值二次项代理", c356),
         ("357", "成交额加权同伴", c357),
         ("358", "成交额加权同伴加换手率与市值二次项代理", c358),
         ("359", "放量加权反转", c359)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
