# -*- coding: utf-8 -*-
"""第六十五批（460–467）+ 第六十六批（468–475）：暴露表解禁后的新信息源。

任务说明"最新状态"第 4 条：每批至少一半必须是新信息源（与 104/390 日度 IC 相关低于 0.9、
单独有效 IC_IR ≥ 0.8）。所以每个新轴都配一个**单独**版本（取证）和一个叠加版本。
第六十五批用 446 的骨架（前一日流通市值 + 申万行业，已错位）：
  460 单独行业内残差阶梯（446 的 ladder 单独跑） / 461 单独日尺度反转 / 462 单独 20 日市值动量反转 /
  463 单独行业内换手拥挤度 / 464–467 分别叠到头部。
第六十六批用 449 的骨架（前一日 10 个风格暴露 + 行业，已错位）：
  468 单独风格分组条件反转 / 469 单独行业滞后收益反转 / 470 单独真换手率相对自身历史 /
  471 单独市值加权行业内上影线残差 / 472–475 分别叠到头部。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
ANCHOR = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n'
UPSH_LINE = '    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'
NEU = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'


def repl(t, old, new, tag, n=1):
    assert t.count(old) == n, (tag, old[:70], t.count(old))
    return t.replace(old, new, 1)


def add_keep(s, cols):
    m = re.search(r'    keep = \[(.*?)\]\n', s, re.S)
    inner = m.group(1)
    for c in cols:
        if '"%s"' % c not in inner:
            inner = inner.rstrip() + ', "%s"' % c
    return s[:m.start(1)] + inner + s[m.end(1):]


def inject(s, code, tag):
    return repl(s, ANCHOR, code + ANCHOR, tag)


def inject_after_upsh(s, code, tag):
    return repl(s, UPSH_LINE, UPSH_LINE + code, tag)


def standalone(s, num, col, extra_keep=()):
    s = add_keep(s, [col] + list(extra_keep))
    s = repl(s, NEU, '    result["factor"] = result["signal"]\n', num + "_raw")
    m = re.search(r'    _cols = \[[^\n]*\n(?:.*?\n)*?    result\["signal"\] = pd\.Series\([^\n]*\)\n', s)
    assert m, num + "_sig"
    return s[:m.start()] + '    result["signal"] = result["%s"]\n' % col + s[m.end():]


def add_slot(s, num, col, wb, wl, extra_keep=()):
    s = add_keep(s, [col] + list(extra_keep))
    m = re.search(r'    _cols = \[([^\n]*)\]\n', s)
    assert m, num + "_cols"
    s = s[:m.start()] + '    _cols = [%s, "%s"]\n' % (m.group(1).rstrip(), col) + s[m.end():]
    m = re.search(r'    _W = np\.tile\(np\.array\(\[([^\]]*)\]\), \(len\(result\), 1\)\)\n', s)
    assert m, num + "_w"
    s = s[:m.start()] + '    _W = np.tile(np.array([%s, %.2f]), (len(result), 1))\n' % (m.group(1).strip(), wb) + s[m.end():]
    m = re.search(r'    _W\[_late\] = np\.array\(\[([^\]]*)\]\)\n', s)
    assert m, num + "_wl"
    s = s[:m.start()] + '    _W[_late] = np.array([%s, %.2f])\n' % (m.group(1).strip(), wl) + s[m.end():]
    return s


# ---------------- 第六十五批（骨架 446：前一日流通市值 + 申万行业） ----------------
AX65 = '''    # ===== 新增轴（前一日暴露表错位）：日尺度面板 =====
    _dly = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first"), _mcap=("mcap", "first"),
        _amt=("sum_amt", "sum"), _ind=("ind_code", "first")).reset_index()
    _dly["_r"] = np.where((_dly["_last"] > 0) & (_dly["_pc"] > 0), np.log(_dly["_last"] / _dly["_pc"]), np.nan)
    _dly["_turn"] = _dly["_amt"] / _dly["_mcap"].where(_dly["_mcap"] > 0)
    _dly = _dly.sort_values(["instrument", "day"]).reset_index(drop=True)
    _g = _dly.groupby("instrument", sort=False)
    _dly["_r_l1"] = _g["_r"].shift(1)
    _dly["_m20"] = _g["_mcap"].shift(20)
    _dly["_turnm20"] = _g["_turn"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    _dly["_mcapmom"] = np.log(_dly["_mcap"] / _dly["_m20"].where(_dly["_m20"] > 0))
    _dly["_imed"] = _dly.groupby(["day", "_ind"], sort=False)["_turn"].transform("median")
    _dly["_turncrowd"] = _dly["_turn"] - _dly["_imed"]
    _mapi = _dly.set_index(["instrument", "day"])
    _key = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["c_drev"] = -_mapi["_r_l1"].reindex(_key).to_numpy()
    df["c_mcapmom"] = -_mapi["_mcapmom"].reindex(_key).to_numpy()
    df["c_turncrowd"] = -_mapi["_turncrowd"].reindex(_key).to_numpy()
'''
NEW65 = ["c_drev", "c_mcapmom", "c_turncrowd"]


def c460(s):   # 单独行业内残差阶梯（446 骨架的 ladder 单独跑）
    s = add_keep(s, ["ladder"])
    s = repl(s, NEU, '    result["factor"] = result["signal"]\n', "460_raw")
    m = re.search(r'    _cols = \[[^\n]*\n(?:.*?\n)*?    result\["signal"\] = pd\.Series\([^\n]*\)\n', s)
    assert m, "460_sig"
    return s[:m.start()] + '    result["signal"] = result["ladder"]\n' + s[m.end():]


def _b65(s, num):
    return inject(add_keep(s, NEW65), AX65, num)


def c461(s):
    return standalone(_b65(s, "461"), "461", "c_drev")


def c462(s):
    return standalone(_b65(s, "462"), "462", "c_mcapmom")


def c463(s):
    return standalone(_b65(s, "463"), "463", "c_turncrowd")


def c464(s):
    return add_slot(_b65(s, "464"), "464", "c_drev", 0.15, 0.10)


def c465(s):
    return add_slot(_b65(s, "465"), "465", "c_mcapmom", 0.10, 0.08)


def c466(s):
    return add_slot(_b65(s, "466"), "466", "c_turncrowd", 0.10, 0.08)


def c467(s):
    s = _b65(s, "467")
    s = add_slot(s, "467a", "c_drev", 0.10, 0.08)
    s = add_slot(s, "467b", "c_mcapmom", 0.08, 0.06)
    return add_slot(s, "467c", "c_turncrowd", 0.08, 0.06)


# ---------------- 第六十六批（骨架 449 + 行业：前一日 10 风格 + 行业） ----------------
IND_PATCH = ('    _ind = _exp["industry_level1_code"].reindex(_midx).astype(str)\n'
             '    df["ind_code"] = _ind.to_numpy()\n')
AX66A = '''    # ===== 新增轴：按前一日风格暴露分组的条件反转（LIQUIDTY 三分位内去均值） =====
    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _q66 = ((df.groupby("date")["s_LIQUIDTY"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)
    df["c_condrev"] = -(_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))
    # ===== 新增轴：真换手率相对自身 20 日历史 =====
    _dly66 = df.groupby(["day", "instrument"], sort=False).agg(
        _mcap=("mcap", "first"), _amt=("sum_amt", "sum"), _ind=("ind_code", "first")).reset_index()
    _dly66["_turn"] = _dly66["_amt"] / _dly66["_mcap"].where(_dly66["_mcap"] > 0)
    _dly66 = _dly66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dly66["_tm20"] = _dly66.groupby("instrument", sort=False)["_turn"].transform(lambda x: x.rolling(20, min_periods=10).mean())
    _dly66["_turnself"] = np.log(_dly66["_turn"] / _dly66["_tm20"].where(_dly66["_tm20"] > 0))
    _m66 = _dly66.set_index(["instrument", "day"])
    _key66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["c_turnself"] = -_m66["_turnself"].reindex(_key66).to_numpy()
'''
AX66B = '''    # ===== 新增轴：行业滞后（前一日）收益反转 =====
    _dly66 = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first"), _ind=("ind_code", "first")).reset_index()
    _dly66["_r"] = np.where((_dly66["_last"] > 0) & (_dly66["_pc"] > 0), np.log(_dly66["_last"] / _dly66["_pc"]), np.nan)
    _idp = _dly66.groupby(["day", "_ind"], sort=False)["_r"].mean().unstack("_ind").sort_index().shift(1)
    _ikey = pd.MultiIndex.from_arrays([df["day"].values, df["ind_code"].values])
    _flat = _idp.stack(dropna=False)
    _flat.index = pd.MultiIndex.from_arrays([_flat.index.get_level_values(0), _flat.index.get_level_values(1)])
    df["c_indrev"] = -_flat.reindex(_ikey).to_numpy()
'''
NEW66 = ["c_condrev", "c_turnself", "c_indrev"]


def _b66(s, num):
    s = repl(s, '_exp_cols = ["float_market_cap", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]',
             '_exp_cols = ["float_market_cap", "industry_level1_code", "SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]', num + "_exp")
    s = repl(s, '    for _c in ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]:\n        df["s_" + _c] = pd.Series(_exp[_c].reindex(_midx).to_numpy(), index=df.index)\n',
             '    for _c in ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]:\n        df["s_" + _c] = pd.Series(_exp[_c].reindex(_midx).to_numpy(), index=df.index)\n' + IND_PATCH, num + "_ind")
    s = inject(add_keep(s, NEW66), AX66A, num + "_a")
    s = inject_after_upsh(s, AX66B, num + "_b")
    return s


def c468(s):
    return standalone(_b66(s, "468"), "468", "c_condrev")


def c469(s):
    return standalone(_b66(s, "469"), "469", "c_indrev")


def c470(s):
    return standalone(_b66(s, "470"), "470", "c_turnself")


def c471(s):
    s = _b66(s, "471")
    s = repl(s, UPSH_LINE, UPSH_LINE + '''    # ===== 新增轴：市值加权行业内上影线残差（前一日流通市值加权） =====
    _wn = df["mcap"] / df["mcap"].groupby([df["date"], df["ind_code"]]).transform("sum").where(lambda x: x > 0)
    df["c_upshres"] = df["c_upsh"] - (df["c_upsh"] * _wn).groupby([df["date"], df["ind_code"]]).transform("sum")
''', "471_upshres")
    return standalone(add_keep(s, ["c_upshres"]), "471", "c_upshres")


def _slots66(s, num, spec):
    for tag, col, wb, wl in spec:
        s = add_slot(s, num + tag, col, wb, wl)
    return s


def c472(s):
    return _slots66(_b66(s, "472"), "472", [("a", "c_condrev", 0.15, 0.10)])


def c473(s):
    return _slots66(_b66(s, "473"), "473", [("a", "c_indrev", 0.10, 0.08)])


def c474(s):
    return _slots66(_b66(s, "474"), "474", [("a", "c_turnself", 0.10, 0.08)])


def c475(s):
    s = _b66(s, "475")
    s = repl(s, UPSH_LINE, UPSH_LINE + '''    _wn = df["mcap"] / df["mcap"].groupby([df["date"], df["ind_code"]]).transform("sum").where(lambda x: x > 0)
    df["c_upshres"] = df["c_upsh"] - (df["c_upsh"] * _wn).groupby([df["date"], df["ind_code"]]).transform("sum")
''', "475_upshres")
    return add_slot(s, "475", "c_upshres", 0.20, 0.10)


CANDS = [("460", "单独行业内残差阶梯", c460),
         ("461", "单独日尺度反转", c461),
         ("462", "单独市值动量二十日", c462),
         ("463", "单独行业内换手拥挤度", c463),
         ("464", "头部加日尺度反转", c464),
         ("465", "头部加市值动量反转", c465),
         ("466", "头部加行业内换手拥挤度", c466),
         ("467", "头部加三新轴", c467),
         ("468", "单独风格条件反转", c468),
         ("469", "单独行业滞后收益反转", c469),
         ("470", "单独真换手率相对自身", c470),
         ("471", "单独市值加权行业内上影残差", c471),
         ("472", "头部加风格条件反转", c472),
         ("473", "头部加行业滞后收益", c473),
         ("474", "头部加真换手率相对自身", c474),
         ("475", "头部加市值加权上影残差", c475)]

B446 = D / "446_行业内部同伴.py"
B449 = D / "449_风格相似度同伴.py"

if __name__ == "__main__":
    for num, name, fn in CANDS:
        base = (B446 if int(num) < 468 else B449).read_text(encoding="utf-8")
        s = fn(base)
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p.name)
