# -*- coding: utf-8 -*-
"""第十批（176+）：把本轮已验证有增量的杠杆叠加（长窗口同伴 / 分钟上影线 / 14:30 首半小时动量 / 新中性化特征）。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, add_rv16, keep, tail, doc  # noqa: E402
from build_wave4 import soft, intraday_peer  # noqa: E402
from build_wave9 import set_lookback  # noqa: E402
from build_wave6 import patch_sql as OHLC_SQL, SNIPS as OHLC_SNIPS, ANCHOR  # noqa: E402
from build_wave5 import add_exposure, add_feats, add_chars  # noqa: E402


def base(k=200, power=3.0, lookback=240, minobs=120, days=400, intra=False):
    s = soft(BASE, k=k, power=power, lookback=lookback, minobs=minobs)
    s = set_lookback(s, days)
    if intra:
        s = intraday_peer(s)
    return s


def add_ohlc(s, cols):
    s = OHLC_SQL(s)
    assert s.count(ANCHOR) == 1
    return s.replace(ANCHOR, ANCHOR + "".join(OHLC_SNIPS[c] for c in cols))


SPECS = [
    ("因子代码/176_长窗口同伴上影线.py", dict(intra=False, ohlc=("c_upsh",), feats=(), chars=(), m30=False,
     rv16=False, wts=[("c_upsh", 0.15)],
     text="""长窗口软同伴（240 日）+ 分钟上影线占比附加（0.15）+ 四特征 1.5 倍中性化（09:30 修复版）。

    叠加两个各自刚验证有效的杠杆：144（同伴相关性窗口 240 日，0.842）与 153（分钟上影线占比，0.834）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。""")),
    ("因子代码/177_长窗口同伴十四点半动量.py", dict(intra=False, ohlc=(), feats=("c_m30",), chars=(), m30=True,
     rv16=False, wts=[("c_m30", 0.15)],
     text="""长窗口软同伴（240 日）+ 14:30 截面专用的首半小时动量（0.15）+ 四特征 1.5 倍中性化（09:30 修复版）。

    叠加 144（0.842）与 147（0.834，日内动量只在尾盘截面生效的机制）。09:30 修复：同 104 号对照版本。""")),
    ("因子代码/178_长窗口同伴上影线六尺度.py", dict(intra=False, ohlc=("c_upsh",), feats=(), chars=(), m30=False,
     rv16=True, wts=[("c_upsh", 0.15)],
     text="""长窗口软同伴（240 日）+ 六尺度分量 + 分钟上影线占比（0.15）+ 四特征 1.5 倍中性化（09:30 修复版）。

    三条已验证杠杆叠加：144（长窗口同伴）、119/142（约 2 日尺度，换手最优）、153（分钟上影线）。
    09:30 修复：同 104 号对照版本。""")),
    ("因子代码/179_长窗口同伴五特征中性.py", dict(intra=False, ohlc=(), feats=("c_turn", "c_mcap2"),
     chars=("c_turn", "c_mcap2"), m30=False, rv16=False, wts=[],
     text="""长窗口软同伴（240 日）+ 六特征 1.5 倍中性化（新增换手率与市值二次项）+（09:30 修复版）。

    148（换手率特征，0.834）与 149（市值二次项，0.834）各自与底座持平，这里同时放进中性化特征集，
      检验两个「官方风格未覆盖的暴露」一起剔除后能否超过 144。
    09:30 修复：同 104 号对照版本。""")),
    ("因子代码/180_日内同伴上影线.py", dict(intra=True, ohlc=("c_upsh",), feats=(), chars=(), m30=False,
     rv16=False, wts=[("c_upsh", 0.15)],
     text="""日内相关同伴（145）+ 分钟上影线占比附加（0.15）+ 四特征 1.5 倍中性化（09:30 修复版）。

    叠加 145（日内共同波动挑同伴，0.838）与 153（分钟上影线，0.834）。09:30 修复：同 104 号对照版本。""")),
    ("因子代码/181_四杠杆叠加.py", dict(intra=False, ohlc=("c_upsh",), feats=("c_m30",), chars=(), m30=True,
     rv16=True, wts=[("c_upsh", 0.15), ("c_m30", 0.15)],
     text="""四杠杆叠加：长窗口同伴 + 六尺度 + 分钟上影线 + 14:30 首半小时动量（09:30 修复版）。

    把本轮四个各自有增量的改动放在一起（权重各 0.15，同伴用 240 日窗口、k=200/3 次方）。
    09:30 修复：同 104 号对照版本。""")),
]

if __name__ == "__main__":
    for path, sp in SPECS:
        s = base(intra=sp["intra"])
        if sp["rv16"]:
            s = add_rv16(s)
        cols = []
        if sp["ohlc"]:
            s = add_ohlc(s, sp["ohlc"])
            cols += list(sp["ohlc"])
        if sp["m30"]:
            s = add_exposure(s) if "exposure_table =" not in s else s
            s = add_feats(s, ("c_m30",))
            cols += ["c_m30"]
        if sp["feats"] and not sp["m30"]:
            s = add_exposure(s)
            s = add_feats(s, sp["feats"])
            cols += list(sp["feats"])
        if cols:
            s = keep(s, tuple(cols))
        if sp["chars"]:
            s = add_chars(s, sp["chars"])
        s = tail(s, tuple(c for c, _ in sp["wts"]), tuple(w for _, w in sp["wts"]))
        s = doc(s, sp["text"])
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
