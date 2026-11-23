# -*- coding: utf-8 -*-
"""第四十六批（307+）：继续换新数据维度——收益-成交额耦合、跳空幅度、波动率的波动、段收益偏度、路径偏离强度。

骨架 = 293（当前头部）。全部 0.15 小权重（偏度 0.10）；09:30 输出 0、末尾 fillna(0.0)。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "293_规模加权同伴幂二EWMA十上影线加平均单笔二日尺度与实体占比.py"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


A_CBODY = '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
A_BODY_SNAP = '            CASE WHEN px_high > 0 AND px_low > 0 AND px_high >= px_low AND px_open > 0 AND px_close > 0 THEN ABS(px_close - px_open) ELSE 0 END AS body_c,\n'
A_DEALS_WIN = '            SUM(deals) AS sum_deals\n        FROM cte_snapshot\n'
A_SEL = '        f.sum_amt, f.sum_vol, f.sum_body, f.last_px, f.pre_close, f.win_high, f.win_low, f.sum_rng, f.sum_upsh, f.sum_deals\n'
A_CAST = '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals"]:\n'
A_KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'


def add(s, code, col, weight, num):
    s = repl(s, A_CBODY, A_CBODY + code, num + "_comp")
    s = repl(s, A_KEEP, A_KEEP.rstrip("\n").replace('"c_body"]', '"c_body", "%s"]' % col) + "\n", num + "_keep")
    s = repl(s, A_SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "%s"], [1.00, 0.30, 0.15, 0.20, 0.15, %.2f])\n' % (col, weight), num + "_sig")
    return s


def c307(s):
    code = ('    # 收益-成交额耦合：40 窗口内 corr(段收益, ln(成交额))，取负（收益放量同向＝知情交易，反转更弱）。\n'
            '    _lr = safe_log_ratio(px, grp["last_px"].shift(1)).where(same_day)\n'
            '    _lv = np.log1p(df["sum_amt"])\n'
            '    _mx, _my = roll(_lr, 40, "mean", 10), roll(_lv, 40, "mean", 10)\n'
            '    _mx2, _my2, _mxy = roll(_lr * _lr, 40, "mean", 10), roll(_lv * _lv, 40, "mean", 10), roll(_lr * _lv, 40, "mean", 10)\n'
            '    _vx = (_mx2 - _mx * _mx).where(lambda v: v > 1e-16)\n'
            '    _vy = (_my2 - _my * _my).where(lambda v: v > 1e-16)\n'
            '    df["c_rvcorr"] = -((_mxy - _mx * _my) / np.sqrt(_vx * _vy)).clip(-1, 1)\n')
    return add(s, code, "c_rvcorr", 0.15, "307")


def c308(s):
    code = ('    # 隔夜跳空幅度：距离昨收的开盘跳空绝对值（在 09:30 段上取），近 40 窗口均值取负。\n'
            '    _gap = safe_log_ratio(df["pre_close"], px).abs().where(same_day, np.nan)\n'
            '    _gap = _gap.where(px > 0)\n'
            '    df["c_gap"] = -roll(_gap, 40, "mean", 10)\n')
    return add(s, code, "c_gap", 0.15, "308")


def c309(s):
    code = ('    # 波动率的波动：|段收益| 的 40 窗口标准差（取负）。\n'
            '    _ar = safe_log_ratio(px, grp["last_px"].shift(1)).where(same_day).abs()\n'
            '    df["c_vov"] = -roll(_ar, 40, "std", 10)\n')
    return add(s, code, "c_vov", 0.15, "309")


def c310(s):
    code = ('    # 段收益偏度（40 窗口，取负）。\n'
            '    _s1 = safe_log_ratio(px, grp["last_px"].shift(1)).where(same_day)\n'
            '    _m1 = roll(_s1, 40, "mean", 10)\n'
            '    _m2 = roll(_s1 * _s1, 40, "mean", 10)\n'
            '    _m3 = roll(_s1 ** 3, 40, "mean", 10)\n'
            '    _var = (_m2 - _m1 * _m1).where(lambda v: v > 1e-16)\n'
            '    df["c_skew"] = -((_m3 - 3 * _m1 * _m2 + 2 * _m1 ** 3) / (_var ** 1.5)).clip(-10, 10)\n')
    return add(s, code, "c_skew", 0.10, "310")


def c311(s):
    code = ('    # 路径偏离强度：|段价−当日VWAP|/段价 的近 40 窗口均值，取负。\n'
            '    _dev = ((px - day_vwap).abs() / px.where(px > 0))\n'
            '    df["c_dev"] = -roll(_dev, 40, "mean", 10)\n')
    return add(s, code, "c_dev", 0.15, "311")



def c312(s):
    cls_snap = ('            CASE WHEN hms = \'150000\' AND vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 '
                'AND amt / vol <= px_high * 1.02 THEN amt ELSE 0 END AS amt_cls_c,\n')
    s = repl(s, A_BODY_SNAP, A_BODY_SNAP + cls_snap, "312_snap")
    s = repl(s, A_DEALS_WIN, '            SUM(amt_cls_c) AS win_cls,\n' + A_DEALS_WIN, "312_win")
    s = repl(s, A_SEL, A_SEL.rstrip("\n") + ", f.win_cls\n", "312_sel")
    s = repl(s, A_CAST, '    for c in ["sum_amt", "sum_vol", "sum_body", "last_px", "pre_close", "win_high", "win_low", "sum_rng", "sum_upsh", "sum_deals", "win_cls"]:\n', "312_cast")
    code = ('    # 昨日收盘集合竞价成交额 / 前 5 个交易日全天成交额均值（整体滞后一天），取负。\n'
            '    _dd2 = df.groupby(["instrument", "day"], sort=False)["sum_amt"].sum().reset_index()\n'
            '    _dd2 = _dd2.sort_values(["instrument", "day"])\n'
            '    _dd2["prev5"] = _dd2.groupby("instrument", sort=False)["sum_amt"].transform(lambda x: x.shift(1).rolling(5, min_periods=3).mean())\n'
            '    _cls_day = df.groupby(["instrument", "day"], sort=False)["win_cls"].sum().rename("v").reset_index().sort_values(["instrument", "day"])\n'
            '    _cls_day["p"] = _cls_day.groupby("instrument", sort=False)["v"].shift(1)\n'
            '    _idx2 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
            '    _pm5 = pd.Series(_dd2.set_index(["instrument", "day"])["prev5"].reindex(_idx2).to_numpy(), index=df.index)\n'
            '    _cl = pd.Series(_cls_day.set_index(["instrument", "day"])["p"].reindex(_idx2).to_numpy(), index=df.index)\n'
            '    df["c_clslag"] = -(_cl / _pm5.where(_pm5 > 0))\n')
    return add(s, code, "c_clslag", 0.15, "312")

CANDS = [("307", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与收益成交额耦合", c307),
         ("308", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与跳空幅度", c308),
         ("309", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与波动率的波动", c309),
         ("310", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与段收益偏度", c310),
         ("311", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与路径偏离强度", c311),
         ("312", "长窗口同伴幂二EWMA十上影线实体平均单笔二日尺度与昨收竞价占比", c312)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
