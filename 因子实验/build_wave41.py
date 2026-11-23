# -*- coding: utf-8 -*-
"""第四十九批（325+）：中性化口径与阶梯内部权重。

325/326/327 把**平台标签残差化用的官方 10 风格 / 32 个行业**拿来做因子自己的中性化（之前只用自造的 6 个特征）。
328/329/330 扫描阶梯内部四个尺度的相对权重（之前一直是等权，从未扫过）。
"""
import pathlib, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "293_规模加权同伴幂二EWMA十上影线加平均单笔二日尺度与实体占比.py"

STYLES = ["SIZE", "BETA", "MOMENTUM", "RESVOL", "SIZENL", "BTOP", "LIQUIDTY", "EARNYILD", "GROWTH", "LEVERAGE"]


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


A_MCAP = '    df["mcap"] = df["mcap"].where(df["mcap"] > 0)\n'
A_CHARS = '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"]\n'
A_FACTOR = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
A_KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_turn", "c_mcap2", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
A_PEERLADDER = '        return rank_blend(cols)\n'

LOAD_STYLES = ('    # 官方 10 风格暴露 + 行业码（与平台标签残差化口径一致），用于因子自身的中性化测试。\n'
               '    _st = dai.query(\n'
               '        "SELECT date, instrument, %s, industry_level1_code FROM %%s" %% exposure_table,\n'
               '        filters={"date": [query_start, end_date]}, compression=True,\n'
               '    ).df()\n'
               '    _st["day"] = pd.to_datetime(_st["date"]).dt.normalize()\n'
               '    _st["instrument"] = _st["instrument"].astype(str)\n'
               '    _st = _st.drop_duplicates(["instrument", "day"])\n'
               '    _sg = _st.set_index(["instrument", "day"])\n'
               '    _sid = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])\n'
               '    for _c in %r:\n'
               '        df["s_" + _c] = pd.Series(_sg[_c].reindex(_sid).to_numpy(), index=df.index)\n'
               '    _ind = _sg["industry_level1_code"].astype(str)\n'
               '    for _c in sorted(_ind.dropna().unique()):\n'
               '        df["i_" + _c] = (_ind.reindex(_sid).to_numpy() == _c).astype(float)\n') % (", ".join(STYLES), STYLES)


def add_styles(s):
    return repl(s, A_MCAP, A_MCAP + LOAD_STYLES, "styles_load")


def set_chars(s, cols, frac, num):
    s = repl(s, A_CHARS, '    CHARS4 = [%s]\n' % ", ".join('"%s"' % c for c in cols), num + "_chars")
    s = repl(s, A_FACTOR, '    result["factor"] = neutralize("signal", CHARS4, %.1f)\n' % frac, num + "_frac")
    return s


def extend_keep(s, cols, num):
    return repl(s, A_KEEP, A_KEEP.rstrip("\n").replace('"c_body"]', '"c_body", %s]' % ", ".join('"%s"' % c for c in cols)) + "\n", num + "_keep")


def set_ladder_weights(s, weights, num):
    new = '        return rank_blend(cols, %s)\n' % ("[" + ", ".join("%.2f" % w for w in weights) + "]")
    return repl(s, A_PEERLADDER, new, num + "_ladder")


# 325：只用官方 10 风格中性化
def c325(s):
    s = add_styles(s)
    s = extend_keep(s, ["s_" + c for c in STYLES], "325")
    return set_chars(s, ["s_" + c for c in STYLES], 1.5, "325")
# 326：自造 6 特征 + 官方 10 风格
def c326(s):
    s = add_styles(s)
    s = extend_keep(s, ["s_" + c for c in STYLES], "326")
    base6 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"]
    return set_chars(s, base6 + ["s_" + c for c in STYLES], 1.5, "326")
# 327：自造 6 特征 + 行业 dummies（行业列名动态，运行时拼进 keep）
def c327(s):
    s = add_styles(s)
    s = repl(s, A_CHARS, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", "c_turn", "c_mcap2"] + sorted([c for c in df.columns if c.startswith("i_")])\n', "327_chars")
    s = repl(s, A_KEEP, A_KEEP + '    keep = keep + sorted([c for c in df.columns if c.startswith("i_")])\n', "327_keep")
    return s


# 328/329/330：阶梯内部权重（r_30, r_rv4, r_rv8, r_day）
def c328(s):
    return set_ladder_weights(s, [1.0, 1.0, 1.0, 0.5], "328")
def c329(s):
    return set_ladder_weights(s, [1.0, 1.0, 1.0, 1.5], "329")
def c330(s):
    return set_ladder_weights(s, [1.0, 1.0, 1.0, 0.01], "330")


CANDS = [("325", "官方风格中性化", c325),
         ("326", "自造特征加官方风格中性化", c326),
         ("327", "自造特征加行业中性化", c327),
         ("328", "阶梯日分量减半", c328),
         ("329", "阶梯日分量加大", c329),
         ("330", "阶梯去掉日分量", c330)]

for num, name, fn in CANDS:
    s = fn(BASE.read_text(encoding="utf-8"))
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(s, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
