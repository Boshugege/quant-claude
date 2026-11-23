# -*- coding: utf-8 -*-
"""第五批（147+）：官方风格之外的中性化特征（用 exposure 的流通市值）、按截面生效的新机制、
以及前几批有效杠杆的再组合。全部含 09:30 修复。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, KEEP_OLD, keep, tail, doc, add_rv16  # noqa: E402
from build_wave4 import soft  # noqa: E402
from build_wave2 import industry_peers  # noqa: E402

EXPOSURE_DEF = ('    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")\n'
                '    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")\n')

DAY_ANCHOR = '    df["day"] = df["date"].dt.normalize()\n'
EXPOSURE_LOAD = '''
    # 官方暴露表（日频）里的流通市值：用于构造「官方风格没有覆盖」的中性化特征。
    _exp = dai.query(
        "SELECT date, instrument, float_market_cap FROM %s" % exposure_table,
        filters={"date": [query_start, end_date]}, compression=True,
    ).df()
    _exp["day"] = pd.to_datetime(_exp["date"]).dt.normalize()
    _exp["instrument"] = _exp["instrument"].astype(str)
    _mcap = _exp.groupby(["instrument", "day"], sort=False)["float_market_cap"].last()
    _midx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["mcap"] = pd.Series(_mcap.reindex(_midx).to_numpy(), index=df.index)
    df["mcap"] = df["mcap"].where(df["mcap"] > 0)
'''

FEAT_ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'
SNIP_TURN = '''
    # 换手率代理：最近 40 个窗口成交额 ÷ 流通市值（官方风格里没有直接的换手率暴露）。
    df["c_turn"] = roll(df["sum_amt"], 40, "sum", 10) / df["mcap"]
'''
SNIP_MCAP2 = '''
    # 流通市值的非线性项：ln(流通市值) 的平方（官方 SIZE 风格只覆盖线性部分）。
    df["c_mcap2"] = np.log(df["mcap"]) ** 2
'''
SNIP_N = '''
    df["_n"] = df.groupby(["instrument", "day"], sort=False).cumcount()
'''

SNIP_M30 = SNIP_N + '''
    # 首半小时收益（10:00 窗口相对 09:30 窗口）：只在 14:30 截面生效（其余截面置缺失）。
    _m = (-df["r_30"]).where(df["_n"] == 1)
    df["c_m30"] = _m.groupby([df["instrument"], df["day"]], sort=False).transform("max")
    df.loc[~((df["date"].dt.hour == 14) & (df["date"].dt.minute == 30)), "c_m30"] = np.nan
'''
SNIP_MIDX = SNIP_N + '''
    # 日内成交重心：当日各窗口成交额的时点重心（窗口序号加权），刻画放量在日内的时间位置，取负。
    _prod = df["sum_amt"] * df["_n"]
    _w = df.groupby(["instrument", "day"], sort=False).transform(lambda x: x.cumsum()) if False else None
    _w = _prod.groupby([df["instrument"], df["day"]], sort=False).cumsum()
    _c = df.groupby(["instrument", "day"], sort=False)["sum_amt"].cumsum()
    df["c_midx"] = -(_w / _c.where(_c > 0))
'''


def add_exposure(src):
    assert src.count(DAY_ANCHOR) == 1
    s = src.replace(DAY_ANCHOR, DAY_ANCHOR + EXPOSURE_LOAD)
    return s.replace('    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")\n',
                     EXPOSURE_DEF)


def add_feats(src, cols):
    body = ""
    if "c_turn" in cols or "c_mcap2" in cols:
        body += SNIP_TURN if "c_turn" in cols else ""
        body += SNIP_MCAP2 if "c_mcap2" in cols else ""
    if "c_m30" in cols:
        body += SNIP_M30
    if "c_midx" in cols:
        body += SNIP_MIDX
    assert src.count(FEAT_ANCHOR) == 1
    return src.replace(FEAT_ANCHOR, FEAT_ANCHOR + body)


def add_chars(src, chars):
    old = '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]'
    assert src.count(old) == 1
    return src.replace(old, '    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx", %s]'
                       % ", ".join('"%s"' % c for c in chars))


SPECS = {
    "因子代码/147_首半小时动量十四点半.py": dict(
        feats=("c_m30",), chars=(), extras=("c_m30",), wts=(0.15,), text=
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 首半小时动量（**只在 14:30 截面生效**，新方向，09:30 修复版）。

    首次使用「按截面限制生效窗口」的机制：把 10:00 窗口相对 09:30 窗口的首半小时收益作为一个独立分量，
      并且只在 14:30 截面给它权重（其余截面置缺失、由 rank_blend 自动把权重归还给阶梯）。
    依据：Gao, Han, Li & Zhou (2018) 与 A 股复现（Chu, Gu & Zhou 2019）都发现首半小时收益能预测尾盘
      （日内动量），而此前的本地粗筛是把首半小时动量全天使用、并因此被否掉（资料库第四节）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/148_换手率中性化.py": dict(
        feats=("c_turn",), chars=("c_turn",), extras=(), wts=(), text=
        """软同伴残差阶梯 + 五特征 1.5 倍中性化（新增换手率特征，新方向，09:30 修复版）。

    在四特征中性化的基础上，用官方暴露表的**流通市值**构造换手率代理（最近 40 窗口成交额 ÷ 流通市值），
      作为一个新的中性化特征。依据：官方 10 个风格里没有直接的换手率暴露，而换手率是 A 股反转因子
      最常见的隐性暴露（高换手 = 情绪/关注度），把它剔掉理论上能提高排序的纯净度。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/149_市值二次项中性化.py": dict(
        feats=("c_mcap2",), chars=("c_mcap2",), extras=(), wts=(), text=
        """软同伴残差阶梯 + 五特征 1.5 倍中性化（新增市值二次项，新方向，09:30 修复版）。

    中性化特征里加入 ln(流通市值)²：官方 SIZE 风格只覆盖市值的**线性**暴露，而反转因子对规模的
      依赖常是非线性的（小盘股的散户结构、壳价值等），二次项能吸收这类弯曲关系。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/150_日内重心附加.py": dict(
        feats=("c_midx",), chars=(), extras=("c_midx",), wts=(0.15,), text=
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 日内成交重心附加（新方向，09:30 修复版）。

    用当日各窗口成交额的时点重心（成交额按窗口序号加权的累计重心）刻画「放量在日内的位置」，
      越接近尾盘说明资金越晚进场，取负。与 124 号（窗口内的分钟重心）是两个尺度上的同一族结构。
    分母只用当日截至当前窗口的累计值，无未来信息。09:30 修复：同 104 号对照版本。"""),
    "因子代码/151_竞价结构组合.py": dict(
        feats=(), chars=(), extras=("c_auc", "c_aucshare"), wts=(0.12, 0.12), text=
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 竞价结构双分量（新方向，09:30 修复版）。

    把 129（竞价相对自身历史的放量）与 130（竞价额相对日常成交规模）按 0.12 / 0.12 权重同时加入，
      两个分量都只依赖 09:25 集合竞价 bar 的成交额与过去 5 个交易日的历史值（严格过去信息）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/152_日内同伴加行业.py": dict(
        feats=(), chars=(), extras=(), wts=(), ind=True, intra=True,
        text="""行业内部 + 日内相关同伴残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    把 132（同伴候选限制在同申万一级行业）与 145（用过去约 20 个交易日 30 分钟收益的相关性挑同伴）
      两个同伴定义改动叠加：既限定同行业，又用日内共同波动来找相似股票。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
}

if __name__ == "__main__":
    for path, sp in SPECS.items():
        s = soft(BASE, k=200, power=3.0, lookback=160 if sp.get("intra") else 60,
                 minobs=60 if sp.get("intra") else 30)
        if sp.get("intra"):
            from build_wave4 import intraday_peer
            s = intraday_peer(s)
        if sp.get("ind"):
            if "exposure_table =" not in s:
                s = s.replace('    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")\n',
                              EXPOSURE_DEF)
            s = industry_peers(s)
        if "c_auc" in sp["extras"] or "c_aucshare" in sp["extras"]:
            from build_wave2 import patch_sql, add_snips
            s = patch_sql(s)
            s = add_snips(s, tuple(sp["extras"]))
        if sp["feats"]:
            s = add_exposure(s)
            s = add_feats(s, sp["feats"])
        cols = tuple(dict.fromkeys(tuple(sp["feats"]) + tuple(sp["extras"])))
        if cols:
            s = keep(s, cols)
        if sp["chars"]:
            s = add_chars(s, sp["chars"])
        if not (sp.get("ind") and not sp["extras"]):
            s = tail(s, sp["extras"], sp["wts"])
        elif not sp["extras"]:
            s = tail(s, (), ())
        s = doc(s, sp["text"])
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
