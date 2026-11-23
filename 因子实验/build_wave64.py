# -*- coding: utf-8 -*-
"""第六十四批（452–459）：为"新信息源"取证 + 把已验证的新轴叠到头部。

任务说明"最新状态"第 4 条要求新信息源"与 104/390 日度 IC 相关低于 0.9、单独有效（单独 IC_IR ≥ 0.8）"，
所以必须有**单独轴**的跑分（第五十五批之后的组合跑分无法证明轴的独立性）。本批：
- 452–455 四个新轴的**单独**版本：已实现偏度 / 五档挂单不平衡(422 方向) / 单笔成交额离散度 / 同伴上一时段收益溢出；
- 456–459 叠加：头部+偏度 0.20、头部+偏度+挂单不平衡、头部+偏度+中性化 2.0、头部+行业中性化+偏度。
"""
import importlib.util, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
NEU = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
SIG390 = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
W390 = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15]), (len(result), 1))\n'
WL390 = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10])\n'
SIG373 = '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"], [1.0, 0.30, 0.15, 0.20, 0.15])\n'


def repl(t, old, new, tag, n=1):
    assert t.count(old) == n, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


def standalone(src, num, col):
    s = pathlib.Path(src).read_text(encoding="utf-8")
    s = repl(s, NEU, '    result["factor"] = result["signal"]\n', num + "_raw")
    # 373 型（rank_blend 单行）与头部型（分槽位权重块）两种骨架都归约成单列信号
    m = re.search(r'    result\["signal"\] = rank_blend\(\[[^\n]*\)\n', s)
    if m:
        s = s[:m.start()] + '    result["signal"] = result["%s"]\n' % col + s[m.end():]
    else:
        m2 = re.search(r'    _cols = \[[^\n]*\n(?:.*?\n)*?    result\["signal"\] = pd\.Series\([^\n]*\)\n', s)
        assert m2, num
        s = s[:m2.start()] + '    result["signal"] = result["%s"]\n' % col + s[m2.end():]
    return s


def add_slot(s, num, col, wb, wl):
    """把 col 追加到头部骨架的分槽位权重里（正则版，可连续追加）。"""
    m = re.search(r'    _cols = \[([^\n]*)\]\n', s)
    assert m, num + "_cols"
    s = s[:m.start()] + '    _cols = [%s, "%s"]\n' % (m.group(1).rstrip(), col) + s[m.end():]
    m = re.search(r'    _W = np\.tile\(np\.array\(\[([^\]]*)\]\), \(len\(result\), 1\)\)\n', s)
    assert m, num + "_w"
    s = s[:m.start()] + '    _W = np.tile(np.array([%s, %.2f]), (len(result), 1))\n' % (m.group(1).strip(), wb) + s[m.end():]
    m = re.search(r'    _W\[_late\] = np\.array\(\[([^\]]*)\]\)\n', s)
    assert m, num + "_wl"
    s = s[:m.start()] + '    _W[_late] = np.array([%s, %.2f])\n' % (m.group(1).strip(), wl) + s[m.end():]
    mk = re.search(r'    keep = \[(.*?)\]\n', s, re.S)
    if mk and '"%s"' % col not in mk.group(1):
        s = s[:mk.start()] + "    keep = [%s]\n" % (mk.group(1).rstrip() + ', "%s"' % col) + s[mk.end():]
    return s


def inject_axis(base, axis_file, num):
    """把 axis_file 里注入的组件代码段与 keep 追加项搬到 base 上。"""
    axis = pathlib.Path(axis_file).read_text(encoding="utf-8")
    m = re.search(r'(    _upsh = df\["sum_upsh".*?\n(?:.*?\n)*?)(?=    # 同伴：|    # 个股 beta|    def )', axis)
    assert m, num + "_axis"
    injected = m.group(1)
    m2 = re.search(r'    keep = \[(.*?)\]\n', axis, re.S)
    m3 = re.search(r'    keep = \[(.*?)\]\n', base, re.S)
    base_keep = m3.group(1)
    for c in re.findall(r'"([a-z0-9_]+)"', m2.group(1)):
        if '"%s"' % c not in base_keep:
            base_keep = base_keep.rstrip() + ', "%s"' % c
    base = base[:m3.start()] + "    keep = [%s]\n" % base_keep + base[m3.end():]
    anchor = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n'
    assert anchor in base, num + "_anchor"
    base = repl(base, anchor, injected, num + "_inject")
    return base


def comb(src390, axes, num):
    """axes: [(文件, 列名, 基础权重, 尾盘权重), ...]"""
    base = pathlib.Path(src390).read_text(encoding="utf-8")
    for f, col, wb, wl in axes:
        base = inject_axis(base, f, num + "_" + col)
        base = add_slot(base, num + "_" + col, col, wb, wl)
    return base


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


W60 = _load("w60", ROOT / "因子实验/build_wave60.py")
A_UPSH = '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n'


def add_sql(s, num, snap, win, names):
    """把 per-minute 列、窗口聚合列、最终 SELECT 与清零循环一并补上（可重复调用）。"""
    s = repl(s, W60.A_BODY_SNAP, W60.A_BODY_SNAP + snap, num + "_snap")
    s = repl(s, W60.A_DEALS_WIN, win + W60.A_DEALS_WIN, num + "_win")
    m = re.search(r'(\n    SELECT\n)(.*?)(\n    FROM cte_window f\n)', s, re.S)
    body = m.group(2).rstrip()
    for c in names:
        if ("f.%s" % c) not in body:
            body += ", f.%s" % c
    s = s[:m.start(2)] + body + "\n" + s[m.end(2):]
    m2 = re.search(r'    for c in \[(.*?)\]:\n', s, re.S)
    inner = m2.group(1).rstrip()
    for c in names:
        if ('"%s"' % c) not in inner:
            inner += ', "%s"' % c
    return s[:m2.start(1)] + inner + s[m2.end(1):]


def apply_skew(s, num):
    s = add_sql(s, num, W60.SK_SNAP, W60.SK_WIN, ("sum_r1sq", "sum_r1cb"))
    code = ('    _sk = df["sum_r1cb"] / (df["sum_r1sq"] ** 1.5).where(df["sum_r1sq"] > 0)\n'
            '    _sk = _sk.where(np.isfinite(_sk))\n'
            '    df["c_skew"] = -_sk.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    return repl(s, A_UPSH, A_UPSH + code, num + "_skew")


def apply_ob(s, num):
    s = repl(s, W60.A_RAW, W60.A_RAW[:-1] + ",\n" + W60.OB_RAW, num + "_raw")
    s = add_sql(s, num, W60.OB_SNAP, W60.OB_WIN, ("sum_bidv", "sum_askv", "sum_bidv1", "sum_askv1"))
    code = ('    _den = (df["sum_bidv"] + df["sum_askv"])\n'
            '    _ob = (df["sum_bidv"] - df["sum_askv"]) / _den.where(_den > 0)\n'
            '    df["c_ob"] = _ob.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=10, min_periods=8).mean())\n')
    return repl(s, A_UPSH, A_UPSH + code, num + "_ob")


B390 = D / "390_上影线EWMA加分槽位权重.py"
B425 = D / "425_分钟收益已实现偏度.py"
B422 = D / "422_五档挂单不平衡反向.py"
B427 = D / "427_单笔成交额离散度.py"
B426 = D / "426_同伴上一时段收益溢出.py"
B444 = D / "444_行业中性化前日行业.py"


def c452(s):
    return standalone(B425, "452", "c_skew")


def c453(s):
    return standalone(B422, "453", "c_ob")


def c454(s):
    return standalone(B427, "454", "c_atdisp")


def c455(s):
    return standalone(B426, "455", "c_spill")


def c456(s):
    return add_slot(apply_skew(s, "456"), "456s", "c_skew", 0.20, 0.10)


def c457(s):
    s = apply_skew(s, "457")
    s = apply_ob(s, "457")
    s = add_slot(s, "457a", "c_skew", 0.15, 0.10)
    return add_slot(s, "457b", "c_ob", 0.15, 0.10)


def c458(s):
    s = add_slot(apply_skew(s, "458"), "458s", "c_skew", 0.15, 0.10)
    return repl(s, NEU, '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n', "458_neu")


def c459(s):
    return add_slot(apply_skew(s, "459"), "459s", "c_skew", 0.15, 0.10)


CANDS = [("452", "单独已实现偏度", c452),
         ("453", "单独五档挂单不平衡", c453),
         ("454", "单独单笔成交额离散度", c454),
         ("455", "单独同伴收益溢出", c455),
         ("456", "头部加偏度二零点二", c456),
         ("457", "头部加偏度与挂单不平衡", c457),
         ("458", "头部加偏度中性化二点零", c458),
         ("459", "行业中性化加偏度", c459)]

if __name__ == "__main__":
    for num, name, fn in CANDS:
        s = fn(B390.read_text(encoding="utf-8"))
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)
