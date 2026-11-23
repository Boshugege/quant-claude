# -*- coding: utf-8 -*-
"""第八批（164+）：围绕「日内相关同伴」这个新高杠杆（145 号代理分 0.838）做定义变体与叠加。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, add_rv16, keep, tail, doc  # noqa: E402
from build_wave4 import soft, intraday_peer  # noqa: E402
from build_micro import SNIP_MAXMIN  # noqa: E402
from build_wave2 import PREV_DAYS, SNIP_AUCSHARE  # noqa: E402

CLEAN = "vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02"
ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

SQL_SNAP_AFTER = "            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,"
SQL_SNAP_ADD = """
            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,"""
SQL_PX_TAIL = """                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,"""
SQL_PX_TAIL_ADD = SQL_PX_TAIL + """
            CASE WHEN %s THEN amt * amt ELSE 0 END AS amt2_c,
            CASE WHEN hms = '092500' AND %s THEN amt ELSE 0 END AS amt_auc,""" % (CLEAN, CLEAN)
SQL_WIN_OLD = "            MAX(pre_close) AS pre_close"
SQL_WIN_ADD = """
            SUM(amt2_c) AS sum_amt2,
            MAX(amt_c) AS max_min_amt,
            SUM(amt_auc) AS win_auc"""
SQL_SEL_OLD = "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close"
SQL_SEL_NEW = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close,\n"
               "        f.sum_amt2, f.max_min_amt, f.win_auc")
SQL_FLT_OLD = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:'
SQL_FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
               '              "sum_amt2", "max_min_amt", "win_auc"]:')


def patch_all(src):
    assert src.count(SQL_SNAP_AFTER) == 1
    src = src.replace(SQL_SNAP_AFTER, SQL_SNAP_AFTER + SQL_SNAP_ADD)
    assert src.count(SQL_PX_TAIL) == 1
    src = src.replace(SQL_PX_TAIL, SQL_PX_TAIL_ADD)
    assert src.count(SQL_WIN_OLD) == 1
    src = src.replace(SQL_WIN_OLD, SQL_WIN_OLD + "," + SQL_WIN_ADD)
    assert src.count(SQL_SEL_OLD) == 1
    src = src.replace(SQL_SEL_OLD, SQL_SEL_NEW)
    assert src.count(SQL_FLT_OLD) == 1
    return src.replace(SQL_FLT_OLD, SQL_FLT_NEW)


SPECS = [
    ("因子代码/164_日内同伴加六尺度.py", dict(rv16=True, comps={}, wts=[], lookback=160, minobs=60,
     text="""日内相关同伴 + 六尺度分量 + 四特征 1.5 倍中性化（09:30 修复版）。

    叠加两个各自验证有效的杠杆：145 号的「用过去约 20 个交易日 30 分钟收益共同波动挑同伴」（代理分 0.838，
      目前最高）与 119 号的约 2 日尺度分量（换手百分位 0.70）。09:30 修复：同 104 号对照版本。""")),
    ("因子代码/165_日内同伴五日窗口.py", dict(rv16=False, comps={}, wts=[], lookback=40, minobs=30,
     text="""日内相关同伴（短窗口，5 个交易日）+ 四特征 1.5 倍中性化（09:30 修复版）。

    同伴相关性只用过去 5 个交易日（40 个 30 分钟观测）估计：更贴近当下资金共同交易的股票，
      但估计噪声更大。与 145 号（20 日）构成窗口长短的对照。09:30 修复：同 104 号对照版本。""")),
    ("因子代码/166_日内同伴六十日窗口.py", dict(rv16=False, comps={}, wts=[], lookback=480, minobs=120,
     text="""日内相关同伴（长窗口，60 个交易日）+ 四特征 1.5 倍中性化（09:30 修复版）。

    与 145 号（20 日）同构，但用 60 个交易日的 30 分钟收益估计相关性（更稳定的共同波动结构）。
    09:30 修复：同 104 号对照版本。""")),
    ("因子代码/167_日内同伴行业六尺度.py", dict(rv16=True, comps={}, wts=[], lookback=160, minobs=60, ind=True,
     text="""日内相关同伴 + 行业内部限制 + 六尺度 + 四特征 1.5 倍中性化（09:30 修复版）。

    把同伴替换的三个改动叠在一起：日内共同波动挑同伴（145）、候选限制在同一申万一级行业内（132）、
      阶梯加入约 2 日尺度（119）。09:30 修复：同 104 号对照版本。""")),
    ("因子代码/168_日内同伴大单冲击.py", dict(rv16=False, comps={"c_maxmin": SNIP_MAXMIN}, wts=[("c_maxmin", 0.15)],
     lookback=160, minobs=60,
     text="""日内相关同伴 + 分钟大单冲击占比附加（0.15）+ 四特征 1.5 倍中性化（09:30 修复版）。

    126 号（单分钟最大成交额占窗口成交额比重）在平台上与底座持平（0.823），这里放到更强的日内同伴底座上再测，
      判断它是被底座覆盖还是本身加了别的信息。09:30 修复：同 104 号对照版本。""")),
    ("因子代码/169_日内同伴竞价占比.py", dict(rv16=False, comps={"c_aucshare": PREV_DAYS + SNIP_AUCSHARE},
     wts=[("c_aucshare", 0.15)], lookback=160, minobs=60,
     text="""日内相关同伴 + 集合竞价额占比附加（0.15）+ 四特征 1.5 倍中性化（09:30 修复版）。

    130 号（竞价成交额 ÷ 过去 5 日全天成交额均值）在旧底座上恰好持平（0.823），放到日内同伴底座上复测；
      竞价数据是 09:25 bar，本身在 09:30 截面即可用，属于严格合规的过去信息。
    09:30 修复：同 104 号对照版本。""")),
]

if __name__ == "__main__":
    for path, sp in SPECS:
        s = soft(BASE, k=200, power=3.0, lookback=sp["lookback"], minobs=sp["minobs"])
        s = intraday_peer(s)
        if sp.get("ind"):
            if "exposure_table =" not in s:
                s = s.replace('    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")\n',
                              '    instruments_table = datasources.get("instruments", "cpt_jyc_2026_instruments")\n'
                              '    exposure_table = datasources.get("exposure", "cpt_jyc_2026_exposure")\n')
            from build_wave2 import industry_peers
            s = industry_peers(s)
        if sp["rv16"]:
            s = add_rv16(s)
        comps = sp["comps"]
        if comps:
            s = patch_all(s)
            assert s.count(ANCHOR) == 1
            s = s.replace(ANCHOR, ANCHOR + "".join(comps.values()))
        cols = tuple(comps.keys())
        if cols or sp["rv16"]:
            s = keep(s, cols)
        s = tail(s, cols, tuple(w for _, w in sp["wts"]))
        s = doc(s, sp["text"])
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
