# -*- coding: utf-8 -*-
"""第四十三批（295+）：在名义最优 293（283 + 实体占比）上做**新数据轴**探索。

295/296 是叠加测试（把两个名义正贡献的杠杆合起来），297-300 是**新方向**：
- 297/298：日内订单流方向（分钟 bar 方向 × 成交额，净主动买卖压力代理）——新数据轴；
- 299：行业内部软同伴（把跨行业相关性屏蔽，132/167 的做法）——新同伴结构；
- 300：日内尾盘成交占比（日级常量，换手零贡献）——新日级轴，用来压换手。

所有候选 09:30 输出 0、末尾 fillna(0.0)，与 104/283 同口径。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE293 = D / "293_规模加权同伴幂二EWMA十上影线加平均单笔二日尺度与实体占比.py"
BASE292 = D / "292_规模加权同伴幂二EWMA十上影线加平均单笔二日尺度与竞价占比.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


# ---------- 各候选的可复用片段 ----------
A_RAW = ('            CAST(COALESCE(amount, 0) AS DOUBLE) AS amt,\n'
         '            CAST(COALESCE(deal_number, 0) AS DOUBLE) AS deals\n')
A_BODY_SNAP = ('            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n')
A_DEALS_WIN = ('            SUM(deals) AS sum_deals\n'
               '        FROM cte_snapshot\n')
A_SEL = ('        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n')
A_CAST = ('    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n')
A_CBODY = ('    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')
A_UPSH10 = ('    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
A_KEEP = ('    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n')
A_SIG = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n')

DIR_SNAP = ('            CASE WHEN vol > 0 AND amt > 0 AND px_close > px_open THEN amt '
            'WHEN vol > 0 AND amt > 0 AND px_close < px_open THEN -amt ELSE 0 END AS dir_amt_c,\n')
DIR_ADD_ROW = ('            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open2,\n')  # 不使用，仅占位说明


def add_dir(s):
    """加入订单流方向（分钟 bar 方向 × 成交额）净额。"""
    s = repl(s, A_RAW, A_RAW.rstrip("\n") + ",\n            CAST(COALESCE(volume, 0) AS DOUBLE) AS vol2\n", "dir_raw")
    # 上面只是确保 vol/amt 已在；dir 计算直接复用已选的 amt/vol/px_close/px_open，无需新列。
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + DIR_SNAP, "dir_snap")
    s = repl(s, A_DEALS_WIN, DIR_ADD_WIN + A_DEALS_WIN, "dir_win")
    s = repl(s, A_SEL, A_SEL.rstrip("\n") + ", f.sum_dir\n", "dir_sel")
    s = repl(s, A_CAST, '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals", "sum_dir"]:\n', "dir_cast")
    s = repl(s, A_CBODY,
             A_CBODY +
             '    # 订单流方向：窗口内 Σ(方向×成交额)/Σ成交额（净主动买卖压力代理），40 窗口均值取负。\n'
             '    _dir = df["sum_dir"] / df["sum_amt"].where(df["sum_amt"] > 0)\n'
             '    df["c_ofi"] = -roll(_dir, 40, "mean", 10)\n', "dir_comp")
    return s


DIR_ADD_WIN = '            SUM(dir_amt_c) AS sum_dir,\n'


def add_body(s):
    """在 292（含竞价占比）基础上补实体占比。"""
    s = repl(s, ('            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,\n'),
             ('            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN px_high - GREATEST(px_open, px_close) ELSE 0 END AS upsh_c,\n'
              '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'), "body_snap")
    s = repl(s, ('            SUM(deals) AS sum_deals\n'), '            SUM(body_c) AS sum_body,\n            SUM(deals) AS sum_deals\n', "body_win")
    s = repl(s, ('        f.sum_amt, f.sum_vol, f.win_auc, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n'),
             '        f.sum_amt, f.sum_vol, f.win_auc, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n', "body_sel")
    s = repl(s, ('    for c in ["sum_amt", "sum_vol", "win_auc", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n'),
             '    for c in ["sum_amt", "sum_vol", "win_auc", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n', "body_cast")
    s = repl(s, ('    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'),
             ('    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'
              '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'), "body_comp")
    s = repl(s, ('    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare"]\n'),
             '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare", "c_body"]\n', "body_keep")
    s = repl(s, ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'),
             '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n', "body_sig")
    return s


def add_industry(s):
    """同伴候选限制在同一申万一级行业内。"""
    s = repl(s, ('    daily_amt = df.groupby(["day", "instrument"], sort=False)["sum_amt"].sum().unstack("instrument").reindex(columns=peer_insts).sort_index()\n'),
             ('    daily_amt = df.groupby(["day", "instrument"], sort=False)["sum_amt"].sum().unstack("instrument").reindex(columns=peer_insts).sort_index()\n'
              '    # 行业内部同伴：用月初之前最近一个交易日的申万一级行业码，把跨行业的相关性屏蔽。\n'
              '    _ind_tbl = dai.query(\n'
              '        "SELECT date, instrument, industry_level1_code FROM %s" % exposure_table,\n'
              '        filters={"date": [query_start, end_date]}, compression=True,\n'
              '    ).df()\n'
              '    _ind_tbl["date"] = pd.to_datetime(_ind_tbl["date"])\n'
              '    _ind_tbl["instrument"] = _ind_tbl["instrument"].astype(str)\n'
              '    ind_wide = _ind_tbl.assign(code=_ind_tbl["industry_level1_code"].astype(str)).pivot_table(\n'
              '        index="date", columns="instrument", values="code", aggfunc="last").reindex(columns=peer_insts).ffill()\n'), "ind_tbl")
    s = repl(s, ('        corr = np.nan_to_num(corr, nan=-1.0)          # 缺失相关性记 -1（等价于不选）\n'
                 '        sub = np.where(ok)[0]\n'),
             ('        corr = np.nan_to_num(corr, nan=-1.0)          # 缺失相关性记 -1（等价于不选）\n'
              '        _pri = ind_wide.index[ind_wide.index < in_month[0]]\n'
              '        if len(_pri):\n'
              '            _row = ind_wide.loc[_pri[-1]].values\n'
              '            if len(_row) == len(ok):\n'
              '                _row = _row[ok]\n'
              '            if len(_row) == len(corr):\n'
              '                _same = (_row[:, None] == _row[None, :]) & pd.notna(_row)[:, None]\n'
              '                corr = np.where(_same, corr, -1.0)\n'
              '        sub = np.where(ok)[0]\n'), "ind_mask")
    return s


def add_late(s):
    """日内尾盘成交占比（日级常量）。"""
    s = repl(s, A_CBODY,
             A_CBODY +
             '    # 日内尾盘成交占比：14:30 段成交额 / 当日全天成交额，取负（尾盘放量 → 情绪高点 → 反转）。日级常量，换手零贡献。\n'
             '    _hday = df["date"].dt.strftime("%H%M")\n'
             '    _daytot = df.groupby(["instrument", "day"], sort=False)["sum_amt"].sum()\n'
             '    _late = df.assign(_h=_hday).query("_h == \'1430\'").groupby(["instrument", "day"], sort=False)["sum_amt"].sum()\n'
             '    df["c_late"] = -(df.set_index(["instrument", "day"]).index.map(_late / _daytot))\n', "late_comp")
    return s


def finish(s, extra_keep, signal_cols, signal_weights, tag):
    keep_old = A_KEEP
    keep_new = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
    if extra_keep:
        keep_new = keep_new.rstrip("\n").replace('"c_body"]', '"c_body", %s]' % ", ".join('"%s"' % c for c in extra_keep)) + "\n"
    s = repl(s, keep_old, keep_new, tag + "_keep")
    sig_new = '    result["signal"] = rank_blend(%s, %s)\n' % (
        "[" + ", ".join('"%s"' % c for c in signal_cols) + "]",
        "[" + ", ".join("%.2f" % w for w in signal_weights) + "]")
    s = repl(s, A_SIG, sig_new, tag + "_sig")
    return s


CANDIDATES = []
# 295 = 293 的 EWMA span 10 → 5
CANDIDATES.append(("295", "规模加权同伴幂二EWMA五上影线实体平均单笔二日尺度",
                   lambda s: repl(s, A_UPSH10, A_UPSH10.replace("span=10", "span=5"), "295")))
# 296 = 292 + 实体占比 0.15
CANDIDATES.append(("296", "规模加权同伴幂二EWMA十上影线实体平均单笔二日尺度与竞价占比",
                   None, "body"))
# 297 = 293 + 订单流方向 0.15
CANDIDATES.append(("297", "规模加权同伴幂二EWMA十上影线实体平均单笔二日尺度与订单流方向",
                   None, "dir", 0.15))
# 298 = 293 + 订单流方向 0.30
CANDIDATES.append(("298", "订单流方向三成", None, "dir", 0.30))
# 299 = 293 + 行业内部同伴
CANDIDATES.append(("299", "行业内部同伴加实体占比", None, "industry"))
# 300 = 293 + 尾盘成交占比 0.10
CANDIDATES.append(("300", "日内尾盘成交占比加实体", None, "late", 0.10))


def build_one(num, name, transform, kind, weight=None):
    if kind == "body":
        s = add_body(BASE292.read_text(encoding="utf-8"))
        s = repl(s, ('    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare", "c_body"]\n'),
                 '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare", "c_body"]\n', "296_keep_ok")
        s = repl(s, ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n'),
                 '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_aucshare", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])\n', "296_sig_ok")
        s = repl(s, '自定义实体占比',
                 '自定义实体占比', "noop") if '自定义实体占比' in s else s
        return s
    s = BASE293.read_text(encoding="utf-8")
    if kind == "dir":
        s = add_dir(s)
        s = finish(s, ["c_ofi"], ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ofi"],
                   [1.0, 0.30, 0.15, 0.20, 0.15, weight], num)
    elif kind == "industry":
        s = add_industry(s)
    elif kind == "late":
        s = add_late(s)
        s = finish(s, ["c_late"], ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_late"],
                   [1.0, 0.30, 0.15, 0.20, 0.15, weight], num)
    if transform is not None:
        s = transform(s)
    return s


for cand in CANDIDATES:
    num, name = cand[0], cand[1]
    transform = cand[2] if len(cand) > 2 else None
    kind = cand[3] if len(cand) > 3 else None
    weight = cand[4] if len(cand) > 4 else None
    s = build_one(num, name, transform, kind, weight)
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
