# -*- coding: utf-8 -*-
"""第十一批（182+）：分钟价格路径与冲击结构的新方向（路径效率、分钟偏度、上涨分钟占比、
价格冲击斜率、单分钟最大涨幅、零变化分钟占比、成交额-波动共同运动）。

全部以当前最强底座（长窗口软同伴 240 日 + 四特征 1.5 倍中性化）为底座，只加一个新方向的附加项（权重 0.15），
便于评分机结果直接读出该方向本身有没有增量。全部含 09:30 修复。
"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import keep, tail, doc  # noqa: E402
from build_wave10 import base  # noqa: E402
from build_micro import add_minute_sql as micro_sql  # noqa: E402

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

MRET2_DN = """            CASE
                WHEN px_open > 0 AND px_close > 0 AND px_close < px_open
                THEN POWER(px_close / px_open - 1, 2) ELSE 0
            END AS mret2_dn,"""

MRET2_DN_ADD = MRET2_DN + """
            CASE WHEN px_open > 0 AND px_close > 0 THEN ABS(px_close / px_open - 1) ELSE 0 END AS amret,
            CASE WHEN px_open > 0 AND px_close > 0 THEN POWER(px_close / px_open - 1, 3) ELSE 0 END AS mret3,
            CASE WHEN px_open > 0 AND px_close > 0 AND px_close > px_open THEN 1 ELSE 0 END AS up_c,
            CASE WHEN px_open > 0 AND px_close > 0 AND px_close < px_open THEN 1 ELSE 0 END AS dn_c,
            CASE WHEN px_open > 0 AND px_close > 0 AND px_close = px_open THEN 1 ELSE 0 END AS zr_c,
            CASE WHEN px_open > 0 AND px_close > 0
                 THEN amt * ABS(px_close / px_open - 1) ELSE 0 END AS amt_aret,"""

WIN_OLD = """            SUM(mret2) AS sum_mret2,
            SUM(mret2_dn) AS sum_mret2_dn"""
WIN_NEW = WIN_OLD + """,
            SUM(mret) AS sum_mret,
            SUM(amret) AS sum_amret,
            SUM(mret3) AS sum_mret3,
            SUM(up_c) AS up_n,
            SUM(dn_c) AS dn_n,
            SUM(zr_c) AS zr_n,
            MAX(mret) AS max_mret,
            SUM(amt_aret) AS sum_amt_aret,
            COUNT(*) AS n_min"""

SEL_OLD = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close,\n"
           "        f.sum_amt2, f.max_min_amt, f.sum_amt_mn, f.sum_mret2, f.sum_mret2_dn")
SEL_NEW = SEL_OLD + (",\n        f.sum_mret, f.sum_amret, f.sum_mret3, f.up_n, f.dn_n, f.zr_n,\n"
                     "        f.max_mret, f.sum_amt_aret, f.n_min")

FLT_OLD = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
           '              "sum_amt2", "max_min_amt", "sum_amt_mn", "sum_mret2", "sum_mret2_dn"]:')
FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
           '              "sum_amt2", "max_min_amt", "sum_amt_mn", "sum_mret2", "sum_mret2_dn",\n'
           '              "sum_mret", "sum_amret", "sum_mret3", "up_n", "dn_n", "zr_n",\n'
           '              "max_mret", "sum_amt_aret", "n_min"]:')

NEW_COLS = ["sum_mret", "sum_amret", "sum_mret3", "up_n", "dn_n", "zr_n",
            "max_mret", "sum_amt_aret", "n_min"]


def add_path_sql(src):
    src = micro_sql(src)                       # 先拿到 px_open / mret / mret2 / sum_amt2 等
    assert src.count(MRET2_DN) == 1
    src = src.replace(MRET2_DN, MRET2_DN_ADD)
    assert src.count(WIN_OLD) == 1
    src = src.replace(WIN_OLD, WIN_NEW)
    assert src.count(SEL_OLD) == 1
    src = src.replace(SEL_OLD, SEL_NEW)
    assert src.count(FLT_OLD) == 1
    return src.replace(FLT_OLD, FLT_NEW)


SNIPS = {
    "c_eff": '''
    # 价格路径效率 = |Σ分钟收益| / Σ|分钟收益|（30 分钟窗口内），最近 40 窗口均值**取负**。
    # 效率低＝价格在同一区间来回震荡（噪声、流动性提供者活跃）；效率高＝单边推进。
    _er = df["sum_mret"].abs() / df["sum_amret"].where(df["sum_amret"] > 0)
    df["c_eff"] = -roll(_er, 40, "mean", 10)
''',
    "c_effpos": '''
    # 路径效率的**正号对照**：其余完全一致，只把符号取正，用来直接定方向。
    _er = df["sum_mret"].abs() / df["sum_amret"].where(df["sum_amret"] > 0)
    df["c_effpos"] = roll(_er, 40, "mean", 10)
''',
    "c_skew": '''
    # 分钟收益的已实现偏度（30 分钟窗口）：Σmret³ / (Σmret²)^1.5，最近 40 窗口均值取负。
    # 此前只在 30 分钟粒度上试过偏度（结论是「粒度太粗」），这里第一次用 1 分钟粒度。
    _sk = df["sum_mret3"] / df["sum_mret2"].pow(1.5).where(df["sum_mret2"] > 0)
    df["c_skew"] = -roll(_sk, 40, "mean", 10)
''',
    "c_updn": '''
    # 上涨分钟占比 = 上涨分钟数 / (上涨+下跌分钟数)，最近 40 窗口均值取负。
    # 与「窗口收益」不同：同样的净收益，可以来自「一直小幅上涨」也可以来自「少数暴涨分钟」，本项专门刻画这一点。
    _p = df["up_n"] / (df["up_n"] + df["dn_n"]).where((df["up_n"] + df["dn_n"]) > 0)
    df["c_updn"] = -roll(_p, 40, "mean", 10)
''',
    "c_kyle": '''
    # 分钟价格冲击斜率（Kyle lambda 的无回归近似）：Σ(成交额 × |分钟收益|) / Σ成交额²，最近 40 窗口均值取正。
    # 含义是「单位成交额推动价格的能力」，是冲击成本的直接度量，与 Amihud 比值（|收益|/成交额）不是同一个量。
    _kl = df["sum_amt_aret"] / df["sum_amt2"].where(df["sum_amt2"] > 0)
    df["c_kyle"] = roll(_kl, 40, "mean", 10)
''',
    "c_maxup": '''
    # 窗口内单分钟最大涨幅，最近 40 窗口均值取负（分钟内被瞬间拉升＝过度反应；与 126 的单分钟最大**成交额**不同）。
    df["c_maxup"] = -roll(df["max_mret"], 40, "mean", 10)
''',
    "c_sparse": '''
    # 零变化分钟占比 = 窗口内「收盘价＝开盘价」的分钟数 / 总分钟数，最近 40 窗口均值取正（成交稀疏＝流动性差）。
    _sp = df["zr_n"] / df["n_min"].where(df["n_min"] > 0)
    df["c_sparse"] = roll(_sp, 40, "mean", 10)
''',
    "c_vvcorr": '''
    # 成交额与波动幅度的共同运动：窗口内 corr(成交额, |分钟收益|)（用和式构造的 Pearson 相关），最近 40 窗口均值取负。
    # 这是「量价同步性」维度，此前所有因子都只用成交额的大小或分布，没有用过它与波动幅度的相关性。
    _n = df["n_min"].where(df["n_min"] > 1)
    _ma = df["sum_amt"] / _n
    _mr = df["sum_amret"] / _n
    _cov = df["sum_amt_aret"] / _n - _ma * _mr
    _sa = (df["sum_amt2"] / _n - _ma ** 2).clip(lower=0) ** 0.5
    _sr = (df["sum_mret2"] / _n - _mr ** 2).clip(lower=0) ** 0.5
    _den = (_sa * _sr).where((_sa * _sr) > 0)
    df["c_vvcorr"] = -roll(_cov / _den, 40, "mean", 10)
''',
}


def add_feature(src, col):
    assert src.count(ANCHOR) == 1
    return src.replace(ANCHOR, ANCHOR + SNIPS[col])


SPECS = [
    ("因子代码/182_分钟路径效率附加.py", "c_eff", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 分钟路径效率附加（新方向，09:30 修复版）。

    新信息：30 分钟窗口内「|Σ分钟收益| ÷ Σ|分钟收益|」的 40 窗口均值（Kaufman 效率比的日内版本），取负。
      同样大小的窗口收益，如果是由来回震荡（低效率）造成的，说明定价噪声大、流动性提供者活跃，
      按 Nagel (2012)「短期反转是提供流动性的报酬」，这类股票之后的反转更该强；单边推进（高效率）则更接近真信息。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/183_分钟路径效率正号对照.py", "c_effpos", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 分钟路径效率（**正号对照**，09:30 修复版）。

    与 182 号只差一个符号，同批提交用来把「效率低反转更强」还是「单边推进反转更强」这个方向问题一次定死。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/184_分钟收益偏度附加.py", "c_skew", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 分钟收益已实现偏度附加（新方向，09:30 修复版）。

    新信息：1 分钟粒度收益的已实现偏度 Σmret³/(Σmret²)^1.5（30 分钟窗口）的 40 窗口均值，取负。
      此前在 30 分钟粒度上试过已实现偏度、结论是「粒度可能太粗」；这里第一次下推到 1 分钟粒度，
      依据是 Amaya et al. (2015) 的有符号跳跃/偏度与彩票偏好。09:30 修复：同 104 号对照版本。"""),
    ("因子代码/185_上涨分钟占比附加.py", "c_updn", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 上涨分钟占比附加（新方向，09:30 修复版）。

    新信息：窗口内上涨分钟数占「上涨+下跌分钟数」的比重，40 窗口均值取负。
      与「窗口收益」正交的部分是分钟级的**一致性**：净收益相同但分钟方向一致的股票，通常是小单持续买入。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/186_分钟价格冲击斜率附加.py", "c_kyle", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + Kyle lambda 附加（新方向，09:30 修复版）。

    新信息：用 Σ(成交额×|分钟收益|)/Σ成交额² 近似「每分钟价格冲击斜率」（Kyle 1985 lambda 的无回归形式），
      40 窗口均值取正。与 Amihud（|收益|÷成交额，已经是 94/95 号的中性化对照项）不同：那是平均比值，
      这里是成交额加权的最小二乘斜率，对「大额分钟推动价格」的敏感度完全不同。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/187_单分钟最大涨幅附加.py", "c_maxup", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 单分钟最大涨幅附加（新方向，09:30 修复版）。

    新信息：窗口内单分钟最大涨幅的 40 窗口均值，取负。126 号用的是单分钟最大**成交额**占比，
      本因子换成单分钟最大**收益**，刻画的是「分钟内被瞬间拉高」的彩票形态（Bali et al. 2011 的日内版本）。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/188_零变化分钟占比附加.py", "c_sparse", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 零变化分钟占比附加（新方向，09:30 修复版）。

    新信息：窗口内「收盘价＝开盘价」的分钟数占比（成交稀疏度）的 40 窗口均值，取正。
      这是**交易活跃度的时间结构**：同样成交额，稀疏成交意味着定价不连续、流动性差。
    09:30 修复：同 104 号对照版本。"""),
    ("因子代码/189_量波共同运动附加.py", "c_vvcorr", """长窗口软同伴（240 日）+ 四特征 1.5 倍中性化 + 成交额-波动共同运动附加（新方向，09:30 修复版）。

    新信息：窗口内 corr(分钟成交额, |分钟收益|) 的 40 窗口均值，取负（量价同步性）。
      此前所有因子只用了成交额的**大小或分布**（集中度、最大占比、相对放量），这是第一次用它与波动幅度的**相关性**。
    09:30 修复：同 104 号对照版本。"""),
]

if __name__ == "__main__":
    for path, col, text in SPECS:
        s = base()
        s = add_path_sql(s)
        s = add_feature(s, col)
        s = keep(s, (col,))
        s = tail(s, (col,), (0.15,))
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
