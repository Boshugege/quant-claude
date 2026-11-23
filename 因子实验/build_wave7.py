# -*- coding: utf-8 -*-
"""第七批（159+）：把各新信号从「相对自身历史」改成「相对同伴」——同伴残差化的新信号族。

伙伴残差是本项目目前最强的杠杆（104/107 起一路到 128 号代理分 0.834）。这里把第六批之前
构造出的各个新信号（相对放量、窗口买卖压力、振幅、竞价放量、成交集中度）都做成同伴残差，
即「个股该信号相对同类股票的偏离」，全部含 09:30 修复。
"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, add_amp, keep, tail, doc  # noqa: E402
from build_wave4 import soft  # noqa: E402
from build_micro import add_minute_sql, SNIPS as MICRO_SNIPS  # noqa: E402
from build_wave2 import patch_sql as W2_SQL, add_snips as W2_SNIPS  # noqa: E402
from build_121_122 import PA_SNIPPET  # noqa: E402

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'
PAT = re.compile(r'    result\["signal"\] = rank_blend\((\[[^\]]*\]), (\[[^\]]*\])\)\n')


def peer_residual(src, col, tag, weight):
    """把 col 分量做成同伴残差（分量 - 同伴均值），并加入排名合成。"""
    res = ('    result["%s"] = result["%s"] - peer_mean("%s").fillna('
           'result.groupby("date")["%s"].transform("mean"))\n') % (tag, col, col, col)
    m = PAT.search(src)
    if m:
        cols = m.group(1)[:-1] + ', "%s"]' % tag
        ws = m.group(2)[:-1] + ', %.2f]' % weight
        out = ('    result["signal"] = rank_blend(%s, %s)\n') % (cols, ws)
        return src[:m.start()] + res + out + src[m.end():]
    old = '    result["factor"] = neutralize("ladder", CHARS4, 1.5)\n'
    assert src.count(old) == 1
    return src.replace(old, res + '    result["signal"] = rank_blend(["ladder", "%s"], [1.0, %.2f])\n' % (tag, weight)
                       + '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n')


def build(cols, tag, weight, text, amp=False, extra_sql=None):
    s = soft(BASE, k=200, power=3.0)
    if amp:
        s = add_amp(s)
    if extra_sql == "micro":
        s = add_minute_sql(s)
        body = "".join(MICRO_SNIPS[c] for c in cols)
    elif extra_sql == "wave2":
        s = W2_SQL(s)
        body = W2_SNIPS(s, tuple(cols))  # 占位，下面统一插入
        body = None
    elif extra_sql == "pa":
        body = ""
    else:
        body = ""
    if extra_sql != "wave2":
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + (PA_SNIPPET if extra_sql == "pa" else body))
    s = keep(s, tuple(cols))
    s = tail(s, (), ())
    s = peer_residual(s, cols[0], tag, weight)
    return doc(s, text)


SPECS = [
    ("因子代码/159_同伴残差相对放量.py", ("c_vshock",), "c_vshock_p", 0.15,
     """同伴残差相对放量 + 残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    把 125 号的「当日累计成交额 ÷ 过去 5 日全天成交额均值」改成**相对同伴**的偏离：个体相对同类股票的
      异常放量才是真正的特质性关注度冲击，整组一起放量的部分（板块情绪）应当剔除（Da, Liu & Schaumburg 2014
      的行业内部反转逻辑，用在量的维度上）。同伴仍用软同伴 k=200 / 3 次方。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/160_同伴残差买卖压力.py", ("c_pa",), "c_pa_p", 0.15,
     """同伴残差买卖压力 + 残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    121 号的窗口买卖压力（窗口末价相对本窗口成交 VWAP 的位置，40 窗口均值）的同伴残差版本：
      只保留个股相对同类股票更强的承接/抛压，去掉整组同向的部分。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/161_同伴残差振幅.py", ("c_amp",), "c_amp_p", 0.15,
     """同伴残差振幅 + 残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    115 号的低振幅附加（20 窗口 (最高-最低)/昨收 均值）的同伴残差版本：相对同类的振幅异常才是
      个股层面的波动冲击，行业共同的波动属于系统性部分。09:30 修复：同 104 号对照版本。"""),
    ("因子代码/162_同伴残差竞价放量.py", ("c_auc",), "c_auc_p", 0.15,
     """同伴残差竞价放量 + 残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    129 号的集合竞价放量（今日竞价成交额 ÷ 过去 5 日竞价成交额均值）的同伴残差版本：
      相对同类股票的竞价异常，才更可能是这只股票自身的信息或情绪，而不是全市场开盘情绪。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/163_同伴残差成交集中度.py", ("c_conc",), "c_conc_p", 0.15,
     """同伴残差成交集中度 + 残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    123 号的分钟成交集中度（窗口内 Σamt²/(Σamt)²）的同伴残差版本：个股相对同类的成交集中度异常，
      刻画的是自身的大单冲击，剔除板块共同的高/低集中度。09:30 修复：同 104 号对照版本。"""),
]


def build_simple(path, cols, tag, weight, text, kind):
    s = soft(BASE, k=200, power=3.0)
    if kind == "amp":
        s = add_amp(s)
    elif kind == "micro":
        s = add_minute_sql(s)
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + "".join(MICRO_SNIPS[c] for c in cols))
    elif kind == "wave2":
        s = W2_SQL(s)
        helper = W2_SNIPS.__wrapped__ if hasattr(W2_SNIPS, "__wrapped__") else None
        s = _w2_add(s, cols)
    elif kind == "pa":
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + PA_SNIPPET)
    s = keep(s, tuple(cols))
    s = tail(s, (), ())
    s = peer_residual(s, cols[0], tag, weight)
    s = doc(s, text)
    pathlib.Path(path).write_text(s, encoding="utf-8")
    print("写出", path, len(s))


_w2_add = None
if __name__ == "__main__":
    from build_wave2 import PREV_DAYS, SNIP_AUC, SNIP_AUCSHARE

    def _w2_add(src, cols):
        body = PREV_DAYS + "".join({"c_auc": SNIP_AUC, "c_aucshare": SNIP_AUCSHARE}[c] for c in cols)
        assert src.count(ANCHOR) == 1
        return src.replace(ANCHOR, ANCHOR + body)

    KINDS = {"159_同伴残差相对放量.py": "micro", "160_同伴残差买卖压力.py": "pa",
             "161_同伴残差振幅.py": "amp", "162_同伴残差竞价放量.py": "wave2",
             "163_同伴残差成交集中度.py": "micro"}
    for path, cols, tag, weight, text in SPECS:
        build_simple(path, cols, tag, weight, text, KINDS[path.split("/")[-1]])
