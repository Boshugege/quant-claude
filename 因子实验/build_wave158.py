# -*- coding: utf-8 -*-
"""第一百五十八批：在 857（条件反转×扩展中性化，IC 均值 0.0799 / 多空 47.05）主臂上继续开轴与诊断。
- 887：+隔夜跳空 0.30（879 的强权重对冲）
- 888：+行业内前一日市值加权同伴残差反转 0.10（新源）
- 889：+行业内条件反转 c_condrev 0.30
- 890：+尾盘 30 分钟动量反转 0.30
- 891：+首窗成交占比 0.30
- 892：中性化仅用 4 特征（诊断扩展中性化贡献）
- 893：中性化 1.75x（2.0 附近的细网格）
- 894：时点重心 0.45 + 隔夜跳空 0.15
"""
import io, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")

def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)

def save(num, name, text):
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write(text)
    print("wrote %d_%s.py" % (num, name))

def sub(t, old, new):
    assert old in t, old[:100]
    return t.replace(old, new, 1)

b857 = load(857)
ANCHOR = '    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]\n'
KEEP_TAIL = '"mcap", "ind_code", "c_utime"]\n'
BLEND = '["c_cond_turnself", "c_utime"], [1.0, 0.15]'

GAP = ('    _dlyg = df.groupby(["day", "instrument"], sort=False).agg(\n'
       '        _first=("last_px", "first"), _pc=("pre_close", "first")).reset_index()\n'
       '    _dlyg["_gap"] = np.where((_dlyg["_first"] > 0) & (_dlyg["_pc"] > 0), np.log(_dlyg["_first"] / _dlyg["_pc"]), np.nan)\n'
       '    _mg = _dlyg.set_index(["instrument", "day"])\n'
       '    _keyg = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
       '    df["c_gap"] = -_mg["_gap"].reindex(_keyg).to_numpy()\n')
INDMW = ('    _dly_b = df.groupby(["day", "instrument"], sort=False).agg(\n'
         '        _last=("last_px", "last"), _pc=("pre_close", "first"), _ind=("ind_code", "first"), _mc=("mcap", "first")).reset_index()\n'
         '    _dly_b["_r"] = np.where((_dly_b["_last"] > 0) & (_dly_b["_pc"] > 0), np.log(_dly_b["_last"] / _dly_b["_pc"]), np.nan)\n'
         '    _wsum = _dly_b.groupby(["day", "_ind"], sort=False)["_mc"].transform("sum")\n'
         '    _dly_b["_wr"] = _dly_b["_r"] * (_dly_b["_mc"] / _wsum.where(_wsum > 0))\n'
         '    _idp = _dly_b.groupby(["day", "_ind"], sort=False)["_wr"].sum().unstack("_ind").sort_index().shift(1)\n'
         '    _flat = _idp.stack(dropna=False)\n'
         '    _ikey = pd.MultiIndex.from_arrays([df["day"].values, df["ind_code"].values])\n'
         '    df["c_indmwrev"] = -_flat.reindex(_ikey).to_numpy()\n')
LATE = ('    _pivW = df.assign(_h=df["date"].dt.strftime("%H%M%S")).pivot_table(index=["instrument", "day"], columns="_h", values="last_px", aggfunc="last")\n'
        '    _keyW = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
        '    _p15 = _pivW["150000"].reindex(_keyW).to_numpy() if "150000" in _pivW.columns else np.full(len(df), np.nan)\n'
        '    _p14 = _pivW["143000"].reindex(_keyW).to_numpy() if "143000" in _pivW.columns else np.full(len(df), np.nan)\n'
        '    _okW = (_p15 > 0) & (_p14 > 0)\n'
        '    df["c_last30"] = -np.where(_okW, np.log(np.where(_okW, _p15, 1.0) / np.where(_okW, _p14, 1.0)), np.nan)\n')
OPEN30 = ('    _hmsW = df["date"].dt.strftime("%H%M%S")\n'
          '    _daW = df.groupby(["instrument", "day"], sort=False)["sum_amt"].transform("sum")\n'
          '    _o1W = df["sum_amt"].where(_hmsW == "093000", 0.0).groupby([df["instrument"], df["day"]], sort=False).transform("sum")\n'
          '    df["c_open30"] = -roll(_o1W / _daW.where(_daW > 0), 40, "mean", 10)\n')

def add_axis(text, code, tag, weight):
    t = sub(text, ANCHOR, code + ANCHOR)
    t = sub(t, KEEP_TAIL, '"mcap", "ind_code", "c_utime", "%s"]\n' % tag)
    t = sub(t, BLEND, '["c_cond_turnself", "c_utime", "%s"], [1.0, 0.15, %s]' % (tag, weight))
    return t

save(887, "条件反转扩展中性化加跳空030", add_axis(b857, GAP, "c_gap", "0.30"))
save(888, "条件反转扩展中性化加行业内市值加权同伴", add_axis(b857, INDMW, "c_indmwrev", "0.10"))
save(889, "条件反转扩展中性化加行业内条件反转", add_axis(b857, '', "c_condrev", "0.30"))
save(890, "条件反转扩展中性化加尾盘动量030", add_axis(b857, LATE, "c_last30", "0.30"))
save(891, "条件反转扩展中性化加首窗成交占比030", add_axis(b857, OPEN30, "c_open30", "0.30"))
save(892, "条件反转扩展中性化四特征诊断", sub(b857, 'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"] + sorted([c for c in result.columns if c.startswith("i_")])', 'CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]'))
save(893, "条件反转扩展中性化一点七五", sub(b857, 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 1.75)'))
save(894, "条件反转扩展中性化加跳空与时点重心045", sub(add_axis(b857, GAP, "c_gap", "0.15"), '["c_cond_turnself", "c_utime", "c_gap"], [1.0, 0.15, 0.15]', '["c_cond_turnself", "c_utime", "c_gap"], [1.0, 0.45, 0.15]'))
