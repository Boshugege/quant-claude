# -*- coding: utf-8 -*-
"""第六十七批（477–484）：再一批新信息源（隔夜跳空 / 市值加权行业动量 / 换手排位变化 / 换手加权上影残差）。

继续按"每个新轴配单独取证版"的结构，全部基于暴露表错位后的骨架（446：前一日流通市值+申万行业；
449：前一日 10 风格暴露+行业）。
"""
import importlib.util, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
ANCHOR = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n'
UPSH_LINE = '    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W65 = _load("w65", ROOT / "因子实验/build_wave65.py")
repl, add_keep, inject, standalone, add_slot = W65.repl, W65.add_keep, W65.inject, W65.standalone, W65.add_slot

AX67 = '''    # ===== 新增轴：隔夜跳空（当日首窗价格 / 昨收）+ 真实换手率排位变化 =====
    _dly = df.groupby(["day", "instrument"], sort=False).agg(
        _first=("last_px", "first"), _pc=("pre_close", "first"), _mcap=("mcap", "first"),
        _amt=("sum_amt", "sum"), _ind=("ind_code", "first")).reset_index()
    _dly["_gap"] = np.where((_dly["_first"] > 0) & (_dly["_pc"] > 0), np.log(_dly["_first"] / _dly["_pc"]), np.nan)
    _dly["_turn"] = _dly["_amt"] / _dly["_mcap"].where(_dly["_mcap"] > 0)
    _dly = _dly.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dly["_trk"] = _dly.groupby("day", sort=False)["_turn"].rank(pct=True)
    _dly["_trk_l1"] = _dly.groupby("instrument", sort=False)["_trk"].shift(1)
    _dly["_trk_chg"] = _dly["_trk"] - _dly["_trk_l1"]
    _m67 = _dly.set_index(["instrument", "day"])
    _key67 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["c_gap"] = -_m67["_gap"].reindex(_key67).to_numpy()
    df["c_turnrkchg"] = -_m67["_trk_chg"].reindex(_key67).to_numpy()
'''
AX67B = '''    # ===== 新增轴：市值加权行业滞后（前一日）收益 =====
    _dly_b = df.groupby(["day", "instrument"], sort=False).agg(
        _last=("last_px", "last"), _pc=("pre_close", "first"), _ind=("ind_code", "first"), _mc=("mcap", "first")).reset_index()
    _dly_b["_r"] = np.where((_dly_b["_last"] > 0) & (_dly_b["_pc"] > 0), np.log(_dly_b["_last"] / _dly_b["_pc"]), np.nan)
    _wsum = _dly_b.groupby(["day", "_ind"], sort=False)["_mc"].transform("sum")
    _dly_b["_wr"] = _dly_b["_r"] * (_dly_b["_mc"] / _wsum.where(_wsum > 0))
    _idp = _dly_b.groupby(["day", "_ind"], sort=False)["_wr"].sum().unstack("_ind").sort_index().shift(1)
    _flat = _idp.stack(dropna=False)
    _ikey = pd.MultiIndex.from_arrays([df["day"].values, df["ind_code"].values])
    df["c_indmwrev"] = -_flat.reindex(_ikey).to_numpy()
    # ===== 新增轴：换手率加权行业内上影线残差 =====
    _tw = df["sum_amt"] / df["mcap"].where(df["mcap"] > 0)
    _twn = _tw / _tw.groupby([df["date"], df["ind_code"]]).transform("sum").where(lambda x: x > 0)
    df["c_upshtwres"] = df["c_upsh"] - (df["c_upsh"] * _twn).groupby([df["date"], df["ind_code"]]).transform("sum")
'''
NEW67 = ["c_gap", "c_turnrkchg", "c_indmwrev", "c_upshtwres"]

B446 = D / "446_行业内部同伴.py"


def _b67(s, num):
    s = inject(add_keep(s, ["c_gap", "c_turnrkchg"]), AX67, num + "_a")
    # AX67B 里的 c_upshtwres 依赖 c_upsh，必须注入在 c_upsh 之后
    s = repl(s, ANCHOR, ANCHOR + '    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n' + AX67B, num + "_b")
    s = add_keep(s, ["c_indmwrev", "c_upshtwres"])
    # 去掉原来重复的 c_upsh 定义行（第二处）
    first = s.index(UPSH_LINE)
    dup = s.find(UPSH_LINE, first + 1)
    assert dup > 0, num + "_dup"
    return s[:dup] + s[dup + len(UPSH_LINE):]


def c477(s):
    return standalone(_b67(s, "477"), "477", "c_gap")


def c478(s):
    return standalone(_b67(s, "478"), "478", "c_indmwrev")


def c479(s):
    return standalone(_b67(s, "479"), "479", "c_turnrkchg")


def c480(s):
    return standalone(_b67(s, "480"), "480", "c_upshtwres")


def c481(s):
    return add_slot(_b67(s, "481"), "481", "c_gap", 0.15, 0.10)


def c482(s):
    return add_slot(_b67(s, "482"), "482", "c_indmwrev", 0.10, 0.08)


def c483(s):
    return add_slot(_b67(s, "483"), "483", "c_turnrkchg", 0.10, 0.08)


def c484(s):
    return add_slot(_b67(s, "484"), "484", "c_upshtwres", 0.20, 0.10)


CANDS = [("477", "单独隔夜跳空", c477),
         ("478", "单独市值加权行业滞后收益", c478),
         ("479", "单独真换手排位变化", c479),
         ("480", "单独换手加权行业内上影残差", c480),
         ("481", "头部加隔夜跳空", c481),
         ("482", "头部加市值加权行业滞后", c482),
         ("483", "头部加真换手排位变化", c483),
         ("484", "头部加换手加权上影残差", c484)]

if __name__ == "__main__":
    for num, name, fn in CANDS:
        s = fn(B446.read_text(encoding="utf-8"))
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p.name)
