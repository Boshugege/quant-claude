# -*- coding: utf-8 -*-
"""第六十三批（444–451）：暴露表解禁后的新信息源（只错一天用），底座 390。

按任务说明"最新状态"第 4 条：每批至少一半是新信息源。本批 6 个新信息源 + 1 个头部探针：
- 444 申万一级行业 dummies 加入中性化特征（情报来源=行业结构，新）；
- 445 真换手率（Σ成交额 / 前一日流通市值）作中性化特征（新）；
- 446 行业内部同伴（同伴集合限定在同一申万一级行业内，相关性加权；新同伴定义）；
- 447 行业内部残差阶梯（4 个尺度在行业内去均值；新）；
- 448 真换手率作信号分量 0.15（新）；
- 449 风格相似度同伴（10 个风格暴露的欧氏距离选最近邻，替代收益相关性；新同伴定义）；
- 450 前一日流通市值加权同伴（规模加权，新）；
- 451 390 中性化力度 2.5（确认实验：探针，私榜风险更高）。

所有暴露字段一律取"严格早于截面日期的最近一条"（前一交易日及更早）。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
B390 = D / "390_上影线EWMA加分槽位权重.py"

A_DAY = '    df = df.sort_values(["instrument", "date"]).reset_index(drop=True)\n    df["day"] = df["date"].dt.normalize()\n'
A_KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body"]\n'
A_CHARS4 = '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]\n'
A_NEU15 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
A_COLS = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_W = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15]), (len(result), 1))\n'
A_WLATE = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10])\n'
A_RV8 = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
A_LADDERDEF = '    def peer_ladder():\n'
A_LADDER = '    result["ladder"] = peer_ladder()\n'
A_LOOP = '    for m in out_months:\n'
A_CORRNAN = '        corr = np.nan_to_num(corr, nan=-1.0)          # 缺失相关性记 -1（等价于不选）\n'
A_AW = '        wts = np.clip(cvals, 0.0, None) ** PEER_POWER   # 软同伴：正相关度加权\n'
STYLE_COLS = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


def add_keep(s, cols, num):
    m = __import__("re").search(r"    keep = \[(.*?)\]\n", s, __import__("re").S)
    assert m, num
    body = m.group(1)
    for c in cols:
        if '"%s"' % c not in body:
            body = body.rstrip() + ', "%s"' % c
    return s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]


MANY = '''
    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")
    # 暴露表错位：第 t 天的市值/风格/行业按当天 15:00 收盘计算，盘中截面不可得；
    # 一律取该股票「严格早于截面日期的最近一条」（前一交易日及更早）。
    _exp_cols = %s
    _exp = dai.query("SELECT date, instrument, " + ", ".join(_exp_cols) + " FROM " + exposure_table,
                     filters={"date": [query_start, end_date]}, compression=True).df()
    _exp["day"] = pd.to_datetime(_exp["date"]).dt.normalize()
    _exp["instrument"] = _exp["instrument"].astype(str)
    _exp = _exp.sort_values(["instrument", "day"]).drop_duplicates(["instrument", "day"]).reset_index(drop=True)
    for _c in _exp_cols:
        _exp[_c] = _exp.groupby("instrument", sort=False)[_c].shift(1)
    _exp = _exp.set_index(["instrument", "day"])
    _midx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["mcap"] = pd.Series(_exp["float_market_cap"].reindex(_midx).to_numpy(), index=df.index)
    df["mcap"] = df["mcap"].where(df["mcap"] > 0)
%s'''


def exp_block(s, num, want_styles=False, want_ind=False, extra=""):
    parts = ['"float_market_cap"']
    if want_styles:
        parts += ['"%s"' % c for c in STYLE_COLS]
    if want_ind:
        parts.append('"industry_level1_code"')
    cols = "[" + ", ".join(parts) + "]"
    tail = extra
    if want_styles:
        tail += ('    for _c in %s:\n'
                 '        df["s_" + _c] = pd.Series(_exp[_c].reindex(_midx).to_numpy(), index=df.index)\n'
                 % repr(STYLE_COLS).replace("'", '"'))
    if want_ind:
        tail += ('    _ind = _exp["industry_level1_code"].reindex(_midx).astype(str)\n'
                 '    df["ind_code"] = _ind.to_numpy()\n'
                 '    for _c in sorted(set(_ind.dropna().unique()) - {"nan", "None"}):\n'
                 '        df["i_" + _c] = (_ind.to_numpy() == _c).astype(float)\n')
    return repl(s, A_DAY, A_DAY + MANY % (cols, tail), num + "_exp")


def neu(s, frac, num):
    return repl(s, A_NEU15, '    result["factor"] = neutralize("signal", CHARS4, %.2f)\n' % frac, num + "_neu")


def slot_add(s, num, col, wb, wl):
    s = repl(s, A_COLS, A_COLS.replace(']', ', "%s"]' % col), num + "_cols")
    s = repl(s, A_W, '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, %.2f]), (len(result), 1))\n' % wb, num + "_w")
    return repl(s, A_WLATE, '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, %.2f])\n' % wl, num + "_wlate")


# ---------- 444 行业 dummies 中性化 ----------
def c444(s):
    s = exp_block(s, "444", want_ind=True)
    s = repl(s, A_KEEP, A_KEEP.rstrip("\n") + ' + sorted([c for c in df.columns if c.startswith("i_")])\n', "444_keep")
    s = repl(s, A_CHARS4, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"] + sorted([c for c in df.columns if c.startswith("i_")])\n', "444_chars")
    return s


# ---------- 445 真换手率作中性化特征 ----------
def c445(s):
    s = exp_block(s, "445")
    code = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
    s = repl(s, A_RV8, A_RV8 + code, "445_code")
    s = add_keep(s, ["c_turn"], "445")
    return repl(s, A_CHARS4, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn"]\n', "445_chars")


# ---------- 446 行业内部同伴 ----------
def c446(s):
    s = exp_block(s, "446", want_ind=True)
    snap = ('    _pf = df[["day", "instrument", "ind_code"]].drop_duplicates(["day", "instrument"]).sort_values(["day", "instrument"])\n'
            '    def _ind_snap(month_start):\n'
            '        prev = _pf[_pf["day"] < month_start].drop_duplicates("instrument", keep="last").set_index("instrument")\n'
            '        return prev["ind_code"].reindex(peer_insts).astype(str).values\n')
    s = repl(s, A_LOOP, snap + A_LOOP, "446_snap")
    s = repl(s, A_CORRNAN,
             A_CORRNAN +
             '        _ind_m = _ind_snap(in_month[0])\n'
             '        corr = np.where(_ind_m[:, None] == _ind_m[None, :], corr, -1.0)   # 同伴限定同行业\n', "446_mask")
    return s


# ---------- 447 行业内部残差阶梯 ----------
def c447(s):
    s = exp_block(s, "447", want_ind=True)
    helper = ('    def peer_ladder_ind():\n'
              '        """阶梯四尺度在申万一级行业内去均值（行业内残差），无行业信息时退化为全截面。"""\n'
              '        _pf = df[["day", "instrument", "ind_code"]].drop_duplicates(["day", "instrument"]).sort_values(["day", "instrument"])\n'
              '        _pf2 = _pf.set_index(["instrument", "day"])["ind_code"].astype(str)\n'
              '        _didx = pd.MultiIndex.from_arrays([result["instrument"].values, result["date"].dt.normalize().values])\n'
              '        result["_ind"] = _pf2.reindex(_didx).to_numpy()\n'
              '        cols = []\n'
              '        for c in ["r_30", "r_rv4", "r_rv8", "r_day"]:\n'
              '            _m = result.groupby(["date", "_ind"], sort=False)[c].transform("mean")\n'
              '            result[c + "_ind"] = result[c] - _m\n'
              '            cols.append(c + "_ind")\n'
              '        return rank_blend(cols)\n')
    s = repl(s, A_LADDERDEF, helper + A_LADDERDEF, "447_helper")
    s = repl(s, A_LADDER, '    result["ladder"] = peer_ladder_ind()\n', "447_ladder")
    s = add_keep(s, ["r_30_ind", "r_rv4_ind", "r_rv8_ind", "r_day_ind"], "447_keep")
    return s


# ---------- 448 真换手率作信号 ----------
def c448(s):
    s = exp_block(s, "448")
    code = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
    s = repl(s, A_RV8, A_RV8 + code, "448_code")
    s = add_keep(s, ["c_turn"], "448")
    return slot_add(s, "448", "c_turn", 0.15, 0.10)


# ---------- 449 风格相似度同伴 ----------
def c449(s):
    s = exp_block(s, "449", want_styles=True)
    snap = ('    _pf = df[["day", "instrument"] + ["s_" + c for c in %s]].drop_duplicates(["day", "instrument"]).sort_values(["day", "instrument"])\n'
            '    def _style_snap(month_start):\n'
            '        prev = _pf[_pf["day"] < month_start].drop_duplicates("instrument", keep="last").set_index("instrument")\n'
            '        v = prev[["s_" + c for c in %s]].reindex(peer_insts).astype(float).values\n'
            '        mu = np.nanmean(v, axis=0); sd = np.nanstd(v, axis=0)\n'
            '        return np.nan_to_num((v - mu) / np.where(sd > 0, sd, 1.0))\n'
            % (repr(STYLE_COLS).replace("'", '"'), repr(STYLE_COLS).replace("'", '"')))
    s = repl(s, A_LOOP, snap + A_LOOP, "449_snap")
    dist = ('        _z = _style_snap(in_month[0])\n'
            '        _D = np.sqrt((( _z[:, None, :] - _z[None, :, :]) ** 2).sum(axis=2))\n'
            '        np.fill_diagonal(_D, np.inf)\n'
            '        _Ds = _D[sub][:, sub]\n'
            '        _top = np.argsort(_Ds, axis=1, kind="stable")[:, :PEER_K]\n'
            '        top = _top\n'
            '        _dv = np.take_along_axis(_Ds, _top, axis=1)\n'
            '        _cvals_d = np.exp(-_dv / (_dv[:, :1] + 1e-9))   # 距离最近权重最大\n')
    s = repl(s, A_CORRNAN, A_CORRNAN + '\n' + dist, "449_dist")
    s = repl(s, '        cvals = corr[np.arange(len(corr))[:, None], top]\n',
             '        cvals = _cvals_d\n', "449_cvals")
    return s


# ---------- 450 前一日流通市值加权同伴 ----------
def c450(s):
    s = exp_block(s, "450")
    snap = ('    _pf = df[["day", "instrument", "mcap"]].drop_duplicates(["day", "instrument"]).sort_values(["day", "instrument"])\n'
            '    def _mcap_snap(month_start):\n'
            '        prev = _pf[_pf["day"] < month_start].drop_duplicates("instrument", keep="last").set_index("instrument")\n'
            '        return prev["mcap"].reindex(peer_insts).fillna(0.0).values\n')
    s = repl(s, A_LOOP, snap + A_LOOP, "450_snap")
    s = repl(s, A_AW,
             '        _aw = _mcap_snap(in_month[0])                # 前一日流通市值加权（大票共同运动信息权重更高）\n'
             '        wts = (np.clip(cvals, 0.0, None) ** PEER_POWER) * _aw[sub][:, None]\n', "450_w")
    return s


CANDS = [("444", "行业中性化前日行业", c444),
         ("445", "真换手率中性化", c445),
         ("446", "行业内部同伴", c446),
         ("447", "行业内部残差阶梯", c447),
         ("448", "真换手率作信号", c448),
         ("449", "风格相似度同伴", c449),
         ("450", "前日市值加权同伴", c450),
         ("451", "头部中性化二点五探针", lambda s: neu(s, 2.5, "451"))]

if __name__ == "__main__":
    for num, name, fn in CANDS:
        s = fn(B390.read_text(encoding="utf-8"))
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)
