# -*- coding: utf-8 -*-
"""第五十批（331+）：把 319 的低相关跳跃轴放进同伴残差框架（看能否恢复独立 IC）、
327 行业中性化头部的组合与强度扫描、以及把线性 rank 混合换成几何混合（新聚合机制）。

331/332/336：同伴残差化的分钟跳跃性（319 的原始轴与 104 相关 0.124，但独立 IC≈0；
            本批检验"同伴残差化"能否把其中的股票特有成分提取出来）。
333：327 头部 + 分钟最大额占比（322 的杠杆）。
334/335：327 头部的行业中性化强度 2.5 / 1.0。
337：把 rank_blend 的线性加权平均换成加权几何平均（新聚合机制）。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "293_规模加权同伴幂二EWMA十上影线加平均单笔二日尺度与实体占比.py"
BASE327 = D / "327_自造特征加行业中性化.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:90], t.count(old))
    return t.replace(old, new, 1)


# ---- 分钟 SQL 扩展（同第四十八批）----
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n'
WIN_COLS = ["avg_absr", "max_absr", "sum_amt2", "max_amt"]
SEL_EXTRA = ", f.avg_absr, f.max_absr, f.sum_amt2, f.max_amt"
CAST_NEW = ('    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals", '
            '"avg_absr", "max_absr", "sum_amt2", "max_amt"]:\n')


def add_minute_sql(s, num):
    snap = ('            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS((px_close - px_open) / px_open) ELSE 0 END AS absr1_c,\n'
            '            CASE WHEN vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02 THEN amt * amt ELSE 0 END AS amt2_c,\n')
    win = ('            AVG(absr1_c) AS avg_absr,\n'
           '            MAX(absr1_c) AS max_absr,\n'
           '            SUM(amt2_c) AS sum_amt2,\n'
           '            MAX(amt_c) AS max_amt,\n')
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, A_DEALS_WIN, win + A_DEALS_WIN, num + "_win")
    s = repl(s, A_SEL, A_SEL.rstrip("\n") + SEL_EXTRA + "\n", num + "_sel")
    s = repl(s, A_CAST, CAST_NEW, num + "_cast")
    return s


A_JUMPCODE = '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
CODE_JUMP = ('    # 分钟跳跃性：窗口内 max|分钟收开| / mean|分钟收开|，近 40 窗口均值取负。\n'
             '    _jmp = df["max_absr"] / df["avg_absr"].where(df["avg_absr"] > 0)\n'
             '    _jmp = _jmp.where(lambda v: (v > 0) & np.isfinite(v))\n'
             '    df["c_jump"] = -roll(_jmp, 40, "mean", 10)\n')
CODE_MAXS = ('    # 分钟最大额占比：窗口内 max amt / Σamt，近 40 窗口均值取负。\n'
             '    _mx = df["max_amt"] / df["sum_amt"].where(df["sum_amt"] > 0)\n'
             '    df["c_maxshare"] = -roll(_mx.where(lambda v: v > 0), 40, "mean", 10)\n')

A_KEEP293 = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_KEEP327 = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_LADDER = '    result["ladder"] = peer_ladder()\n'
A_SIG293 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'
A_FACTOR15 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'


def add_keep(s, cols, num):
    add = "".join(', "%s"' % c for c in cols)
    return repl(s, A_KEEP293, A_KEEP293.rstrip("\n").replace('"c_body"]', '"c_body"%s]' % add) + "\n", num + "_keep")


def add_jump_code(s, num):
    return repl(s, A_JUMPCODE, A_JUMPCODE + CODE_JUMP, num + "_jumpcode")


def add_maxs_code(s, num):
    return repl(s, A_JUMPCODE, A_JUMPCODE + CODE_MAXS, num + "_maxscode")


# 331/332：同伴残差跳跃性
def resid_jump(s, num, w):
    s = add_minute_sql(s, num)
    s = add_jump_code(s, num)
    s = add_keep(s, ["c_jump"], num)
    resid = ('    result["c_jump_p"] = result["c_jump"] - peer_mean("c_jump").fillna(result.groupby("date")["c_jump"].transform("mean"))\n')
    s = repl(s, A_LADDER, A_LADDER + resid, num + "_resid")
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_jump_p"], [1.0, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % w
    return repl(s, A_SIG293, sig, num + "_sig")


# 333：327 头部 + 分钟最大额占比
def c333(s):
    s = add_minute_sql(s, "333")
    s = add_maxs_code(s, "333")
    s = repl(s, A_KEEP327, A_KEEP327.rstrip("\n").replace('"c_body"]', '"c_body", "c_maxshare"]') + "\n", "333_keep")
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_maxshare"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n'
    return repl(s, A_SIG293, sig, "333_sig")


# 334/335：327 头部的行业中性化强度
def set_frac327(s, frac, num):
    return repl(s, A_FACTOR15, '    result["factor"] = neutralize("signal", CHARS4, %.1f)\n' % frac, num + "_frac")


# 336：同伴残差跳跃性 + 分钟最大额占比
def c336(s):
    s = add_minute_sql(s, "336")
    s = add_jump_code(s, "336")
    s = add_maxs_code(s, "336")
    s = add_keep(s, ["c_jump", "c_maxshare"], "336")
    resid = ('    result["c_jump_p"] = result["c_jump"] - peer_mean("c_jump").fillna(result.groupby("date")["c_jump"].transform("mean"))\n')
    s = repl(s, A_LADDER, A_LADDER + resid, "336_resid")
    sig = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_jump_p", "c_maxshare"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.10, 0.15])\n'
    return repl(s, A_SIG293, sig, "336_sig")


# 337：几何混合聚合
A_RB = ('        num = np.nansum(np.nan_to_num(ranks) * w, axis=1)\n'
        '        return pd.Series(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan), index=result.index)\n')
RB_GEO = ('        rp = np.where(has, ranks + 0.5, np.nan)\n'
          '        logr = np.log(np.where(rp > 0, rp, 1.0))\n'
          '        num = np.nansum(np.where(has, logr, 0.0) * w, axis=1)\n'
          '        return pd.Series(np.where(den > 0, np.exp(num / np.where(den > 0, den, 1.0)) - 0.5, np.nan), index=result.index)\n')


def c337(s):
    return repl(s, A_RB, RB_GEO, "337_geo")


CANDS = [("331", "同伴残差分钟跳跃性0.10", lambda s: resid_jump(s, "331", 0.10)),
         ("332", "同伴残差分钟跳跃性0.25", lambda s: resid_jump(s, "332", 0.25)),
         ("333", "行业中性化加分钟最大额占比", c333),
         ("334", "行业中性化强度2.5", lambda s: set_frac327(s, 2.5, "334")),
         ("335", "行业中性化强度1.0", lambda s: set_frac327(s, 1.0, "335")),
         ("336", "同伴残差跳跃性加分钟最大额占比", c336),
         ("337", "几何混合聚合", c337)]

for num, name, fn in CANDS:
    src = BASE327 if num in ("333", "334", "335") else BASE
    s = fn(src.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
