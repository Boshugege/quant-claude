# -*- coding: utf-8 -*-
"""第五十三批（366+）：**新方向优先**（不为缓存复用旧思路）。

- 366/367：同伴残差化的分钟跳跃性（331 在旧家族里显示残差化后不再稀释，合法重测）。
- 368：30 分钟同伴相关（个股收益与软同伴组收益的窗口相关，拥挤/特质轴）。
- 369：首半小时动量（Gao/Han 2018、Chu 2019；A 股日内动量）作为后段截面的独立分量。
- 370：同时段异常放量（当日窗口成交量 ÷ 该时段过去 20 日均值）。
- 371：分槽位差异化权重（尾盘降低快速分量、提高阶梯权重）。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "353_同伴上影线平均单笔二日尺度与实体占比.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
    return t.replace(old, new, 1)


import re
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
A_LADDER = '    result["ladder"] = peer_ladder()\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'

# ---- 分钟级 SQL 扩展（同第四十八批）----
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close, f.win_high, f.win_low,\n        f.sum_rng, f.sum_upsh, f.sum_dnsh, f.sum_body, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "win_high", "win_low",\n              "sum_rng", "sum_upsh", "sum_dnsh", "sum_body", "sum_deals"]:\n'


def add_minute_sql(s, num):
    snap = ('            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr1_c,\n')
    win = ('            AVG(absr1_c) AS avg_absr,\n'
           '            MAX(absr1_c) AS max_absr,\n')
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.replace("f.sum_deals\n", "f.sum_deals, f.avg_absr, f.max_absr\n"), num + "_sel")
    s = repl(s, A_CAST, A_CAST.replace('"sum_deals"]', '"sum_deals", "avg_absr", "max_absr"]'), num + "_cast")
    return s


CODE_JUMP = ('    # 分钟跳跃性：窗口内 max|分钟收开| / mean|分钟收开|，近 40 窗口均值取负。\n'
             '    _jmp = df["max_absr"] / df["avg_absr"].where(df["avg_absr"] > 0)\n'
             '    _jmp = _jmp.where(lambda v: (v > 0) & np.isfinite(v))\n'
             '    df["c_jump"] = -roll(_jmp, 40, "mean", 10)\n')
RESID = '    result["c_jump_p"] = result["c_jump"] - peer_mean("c_jump").fillna(result.groupby("date")["c_jump"].transform("mean"))\n'


def resid_jump(s, num, w):
    s = add_minute_sql(s, num)
    s = repl(s, A_UPSH, A_UPSH + CODE_JUMP, num + "_jump")
    s = add_keep(s, ["c_jump"], num)
    s = repl(s, A_LADDER, A_LADDER + RESID, num + "_resid")
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_jump_p"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % w)
    return repl(s, SIG353, sig, num + "_sig")


def c366(s):
    return resid_jump(s, "366", 0.10)
def c367(s):
    return resid_jump(s, "367", 0.25)


def c368(s):
    """30 分钟同伴相关：个股 30 分钟收益与软同伴组同截面均值收益的 40 窗口相关。"""
    s = repl(s, '    mkt30 = ret30.groupby(df["date"]).transform("mean")\n',
             '    mkt30 = ret30.groupby(df["date"]).transform("mean")\n    df["ret30"] = ret30\n', "368_ret30")
    s = add_keep(s, ["ret30"], "368")
    helper = ('    def roll_r(s_, k, fn, mp):\n'
              '        return s_.groupby(result["instrument"], sort=False).transform(lambda x: getattr(x.rolling(k, min_periods=mp), fn)())\n'
              '    _pm = peer_mean("ret30")\n'
              '    _x = result["ret30"]\n'
              '    _ex, _ey = roll_r(_x, 40, "mean", 10), roll_r(_pm, 40, "mean", 10)\n'
              '    _exy = roll_r(_x * _pm, 40, "mean", 10)\n'
              '    _vx = (roll_r(_x * _x, 40, "mean", 10) - _ex ** 2).where(lambda v: v > 1e-12)\n'
              '    _vy = (roll_r(_pm * _pm, 40, "mean", 10) - _ey ** 2).where(lambda v: v > 1e-12)\n'
              '    result["c_peercorr"] = ((_exy - _ex * _ey) / np.sqrt(_vx * _vy)).clip(-1, 1)\n')
    s = repl(s, A_LADDER, A_LADDER + helper, "368_corr")
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_peercorr"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n')
    return repl(s, SIG353, sig, "368_sig")


def c369(s):
    """首半小时动量：+log(09:30 价/昨收) = −r_30(09:30)，映射到当日各截面（Gao 2018 日内动量）。"""
    helper = ('    result["day"] = result["date"].dt.normalize()\n'
              '    _slot = result["date"].dt.strftime("%H%M")\n'
              '    _idx = pd.MultiIndex.from_arrays([result["day"], result["instrument"]])\n'
              '    _fh = result.loc[_slot == "0930"].set_index(["day", "instrument"])["r_30"]\n'
              '    result["c_fh"] = -_fh.reindex(_idx).to_numpy()\n')
    s = repl(s, A_LADDER, A_LADDER + helper, "369_fh")
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_fh"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n')
    return repl(s, SIG353, sig, "369_sig")


def c370(s):
    """同时段异常放量：当日窗口成交量 ÷ 该时段过去 20 日均值（前一日截止）。"""
    s = repl(s, A_RV8, A_RV8 + ('    # 同时段异常放量：窗口成交量 ÷ 该时段过去 20 日均值（shift 1，不含当日）。\n'
                                '    df["slot"] = df["date"].dt.strftime("%H%M")\n'
                                '    _base = df.groupby(["instrument", "slot"], sort=False)["sum_vol"].transform(\n'
                                '        lambda x: x.rolling(20, min_periods=8).mean().shift(1))\n'
                                '    df["c_absvol"] = df["sum_vol"] / _base.where(_base > 0)\n'), "370_absvol")
    s = add_keep(s, ["sum_vol", "c_absvol"], "370")
    sig = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_absvol"], '
           '[1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n')
    return repl(s, SIG353, sig, "370_sig")


def c371(s):
    """分槽位差异化权重：14:00/14:30 降低快速分量（上影线/实体），提高阶梯权重。"""
    helper = ('    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
              '    _rk = (result.groupby("date")[_cols].rank(pct=True) - 0.5).to_numpy()\n'
              '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15]), (len(result), 1))\n'
              '    _late = result["date"].dt.strftime("%H%M").isin(["1400", "1430"]).to_numpy()\n'
              '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10])\n'
              '    _has = ~np.isnan(_rk)\n'
              '    _den = (_has * _W).sum(axis=1)\n'
              '    _num = np.nansum(np.nan_to_num(_rk) * _W, axis=1)\n'
              '    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)\n')
    return repl(s, SIG353, helper, "371_sig")


CANDS = [("366", "同伴残差跳跃性010", c366),
         ("367", "同伴残差跳跃性025", c367),
         ("368", "三十分钟同伴相关", c368),
         ("369", "首半小时动量", c369),
         ("370", "同时段异常放量", c370),
         ("371", "分槽位差异化权重", c371)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
