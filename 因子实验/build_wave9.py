# -*- coding: utf-8 -*-
"""第九批（170+）：同伴相关性历史长度（144 号 240 日窗口代理分 0.842 为本轮最高）的边界与叠加。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, add_rv16, keep, tail, doc  # noqa: E402
from build_wave4 import soft, intraday_peer  # noqa: E402
from build_wave8 import patch_all, ANCHOR  # noqa: E402
from build_micro import SNIP_MAXMIN  # noqa: E402

LOOKBACK_OLD = "    LOOKBACK_DAYS = 150"


def set_lookback(src, days):
    assert src.count(LOOKBACK_OLD) == 1
    return src.replace(LOOKBACK_OLD, "    LOOKBACK_DAYS = %d" % days)


def sec(text):
    return doc


SPECS = [
    ("因子代码/170_同伴窗口240日.py", dict(k=200, power=3.0, lookback=240, minobs=120, days=400, rv16=False,
     comps={}, wts=[], intra=False,
     text="""软同伴（k=200/3 次方）+ 240 日相关性窗口 + 残差阶梯 + 四特征 1.5 倍中性化（09:30 修复版）。

    144 号把同伴相关性窗口从 60 日改成 240 日，代理分 0.842（本轮最高），说明「更长的相似性结构」本身是杠杆。
      本因子把向前取数区间从 150 天放大到 400 天，确保 240 个交易日的窗口真的能取满（原版受取数窗口限制，
      实际只用到约 150 天）。09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。""")),
    ("因子代码/171_同伴窗口480日.py", dict(k=200, power=3.0, lookback=480, minobs=240, days=650, rv16=False,
     comps={}, wts=[], intra=False,
     text="""软同伴 + 480 日相关性窗口 + 残差阶梯 + 四特征 1.5 倍中性化（09:30 修复版）。

    窗口继续外推到 480 个交易日（约两年），并把向前取数放大到 650 天。用于确认「窗口越长越好」是否走到顶。
    09:30 修复：同 104 号对照版本。""")),
    ("因子代码/172_日内同伴240日.py", dict(k=200, power=3.0, lookback=1920, minobs=400, days=400, rv16=False,
     comps={}, wts=[], intra=True,
     text="""日内相关同伴（30 分钟收益、240 个交易日窗口）+ 四特征 1.5 倍中性化（09:30 修复版）。

    把 145 号（日内共同波动挑同伴，20 日窗口，代理分 0.838）的估计窗口拉长到 240 个交易日（1920 个 30 分钟观测），
      并同步放大向前取数区间。用于检验「日内同伴 + 长窗口」是否叠加。09:30 修复：同 104 号对照版本。""")),
    ("因子代码/173_长窗口同伴六尺度.py", dict(k=200, power=3.0, lookback=240, minobs=120, days=400, rv16=True,
     comps={}, wts=[], intra=False,
     text="""长窗口软同伴（240 日）+ 六尺度分量 + 四特征 1.5 倍中性化（09:30 修复版）。

    叠加两个已分别验证有效的杠杆：240 日同伴窗口（144 号，0.842）与约 2 日尺度分量（119 号，换手最优）。
    09:30 修复：同 104 号对照版本。""")),
    ("因子代码/174_长窗口同伴大单冲击.py", dict(k=200, power=3.0, lookback=240, minobs=120, days=400, rv16=False,
     comps={"c_maxmin": SNIP_MAXMIN}, wts=[("c_maxmin", 0.15)], intra=False,
     text="""长窗口软同伴（240 日）+ 分钟大单冲击占比附加（0.15）+ 四特征 1.5 倍中性化（09:30 修复版）。

    126 号（大单冲击）在旧底座上与底座持平，这里在更强的新底座上复测；权重仍是 0.15。
    09:30 修复：同 104 号对照版本。""")),
    ("因子代码/175_长窗口同伴k400.py", dict(k=400, power=4.0, lookback=240, minobs=120, days=400, rv16=False,
     comps={}, wts=[], intra=False,
     text="""长窗口软同伴（240 日）+ k=400、相关度 4 次方 + 四特征 1.5 倍中性化（09:30 修复版）。

    140 号（k=400/4 次方、60 日窗口）比 k=200/3 次方略低，这里在长窗口底座上复核同伴软度与窗口长度的交互。
    09:30 修复：同 104 号对照版本。""")),
]

if __name__ == "__main__":
    for path, sp in SPECS:
        s = soft(BASE, k=sp["k"], power=sp["power"], lookback=sp["lookback"], minobs=sp["minobs"])
        s = set_lookback(s, sp["days"])
        if sp["intra"]:
            s = intraday_peer(s)
        if sp["rv16"]:
            s = add_rv16(s)
        if sp["comps"]:
            s = patch_all(s)
            assert s.count(ANCHOR) == 1
            s = s.replace(ANCHOR, ANCHOR + "".join(sp["comps"].values()))
        cols = tuple(sp["comps"].keys())
        if cols or sp["rv16"]:
            s = keep(s, cols)
        s = tail(s, cols, tuple(w for _, w in sp["wts"]))
        s = doc(s, sp["text"])
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
