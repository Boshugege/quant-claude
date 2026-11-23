# -*- coding: utf-8 -*-
"""第二批新方向（129+）：集合竞价结构、分钟净买压、行业内部同伴，以及两个独立探针。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, soft_peer, keep, tail, doc  # noqa: E402
from build_micro import SQL_SNAP_AFTER, SQL_SNAP_ADD, SNAP_SEL_OLD, SNAP_SEL_NEW, SQL_WIN_OLD  # noqa: E402

CLEAN = "vol > 0 AND amt > 0 AND px_low > 0 AND amt / vol >= px_low * 0.98 AND amt / vol <= px_high * 1.02"

SQL_PX_TAIL = """                WHEN px_close > 0 THEN px_close
                ELSE 0
            END AS px,"""
SQL_PX_TAIL_ADD = SQL_PX_TAIL + """
            CASE
                WHEN hms = '092500' AND %s THEN amt ELSE 0
            END AS amt_auc,
            CASE
                WHEN %s AND px_open > 0 AND px_close > 0 AND px_close > px_open THEN amt
                WHEN %s AND px_open > 0 AND px_close > 0 AND px_close < px_open THEN -amt
                ELSE 0
            END AS samt,""" % (CLEAN, CLEAN, CLEAN)

SQL_WIN_ADD = """
            SUM(amt_auc) AS win_auc,
            SUM(samt) AS sum_samt"""

SQL_SEL_OLD = "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close"
SQL_SEL_NEW = ("        f.sum_amt, f.sum_vol, f.last_px, f.pre_close,\n"
               "        f.win_auc, f.sum_samt")
SQL_FLT_OLD = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:'
SQL_FLT_NEW = ('    for c in ["sum_amt", "sum_vol", "last_px", "pre_close",\n'
               '              "win_auc", "sum_samt"]:')

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

PREV_DAYS = '''
    def _prev_days(col, k):
        """按 (instrument, day) 汇总后向前取 k 个交易日均值（只用过去数据），对齐回每一行。"""
        g = df.groupby(["instrument", "day"], sort=False)[col].sum().rename("v").reset_index()
        g = g.sort_values(["instrument", "day"])
        g["p"] = g.groupby("instrument", sort=False)["v"].transform(
            lambda x: x.shift(1).rolling(k, min_periods=3).mean())
        m = g.set_index(["instrument", "day"])["p"]
        idx = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
        return pd.Series(m.reindex(idx).to_numpy(), index=df.index)

'''

SNIP_AUC = '''
    # 集合竞价放量：今日 09:25 竞价成交额 / 过去 5 个交易日竞价成交额均值（只用过去数据），取负。
    _auc_today = df.groupby(["instrument", "day"], sort=False)["win_auc"].transform("sum")
    _auc_prev = _prev_days("win_auc", 5)
    df["c_auc"] = -(_auc_today / _auc_prev.where(_auc_prev > 0))
'''

SNIP_AUCSHARE = '''
    # 竞价相对成交：今日 09:25 竞价成交额 / 过去 5 个交易日全天成交额均值，取负。
    _auc_today = df.groupby(["instrument", "day"], sort=False)["win_auc"].transform("sum")
    _day_prev = _prev_days("sum_amt", 5)
    df["c_aucshare"] = -(_auc_today / _day_prev.where(_day_prev > 0))
'''

SNIP_NETFLOW = '''
    # 分钟净买压：窗口内(涨分钟成交额 - 跌分钟成交额) / 窗口成交额，最近 40 窗口均值取负。
    _net = df["sum_samt"] / df["sum_amt"].where(df["sum_amt"] > 0)
    df["c_netflow"] = -roll(_net, 40, "mean", 10)
'''


def patch_sql(src):
    assert src.count(SQL_SNAP_AFTER) == 1
    src = src.replace(SQL_SNAP_AFTER, SQL_SNAP_AFTER + SQL_SNAP_ADD)
    assert src.count(SNAP_SEL_OLD) == 1
    src = src.replace(SNAP_SEL_OLD, SNAP_SEL_NEW)
    assert src.count(SQL_PX_TAIL) == 1
    src = src.replace(SQL_PX_TAIL, SQL_PX_TAIL_ADD)
    assert src.count(SQL_WIN_OLD) == 1
    src = src.replace(SQL_WIN_OLD, SQL_WIN_OLD + "," + SQL_WIN_ADD)
    assert src.count(SQL_SEL_OLD) == 1
    src = src.replace(SQL_SEL_OLD, SQL_SEL_NEW)
    assert src.count(SQL_FLT_OLD) == 1
    return src.replace(SQL_FLT_OLD, SQL_FLT_NEW)


def add_snips(src, cols):
    body = PREV_DAYS if any(c in ("c_auc", "c_aucshare") for c in cols) else ""
    for c in cols:
        body += {"c_auc": SNIP_AUC, "c_aucshare": SNIP_AUCSHARE, "c_netflow": SNIP_NETFLOW}[c]
    return src.replace(ANCHOR, ANCHOR + body)


def probe_tail(src, col):
    old = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["factor"] = neutralize("ladder", CHARS4, 1.5)'''
    assert src.count(old) == 1
    new = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["factor"] = neutralize("%s", CHARS4, 1.5)''' % col
    return src.replace(old, new)


IND_SQL = '''
    # 行业内部同伴：用月初之前最近一个交易日的申万一级行业码，把跨行业的相关性屏蔽。
    _ind_tbl = dai.query(
        "SELECT date, instrument, industry_level1_code FROM %s" % exposure_table,
        filters={"date": [query_start, end_date]}, compression=True,
    ).df()
    _ind_tbl["date"] = pd.to_datetime(_ind_tbl["date"])
    _ind_tbl["instrument"] = _ind_tbl["instrument"].astype(str)
    ind_wide = _ind_tbl.assign(code=_ind_tbl["industry_level1_code"].astype(str)).pivot_table(
        index="date", columns="instrument", values="code", aggfunc="last").reindex(columns=peer_insts).ffill()
'''

IND_MASK = '''
        _pri = ind_wide.index[ind_wide.index < in_month[0]]
        if len(_pri):
            _row = ind_wide.loc[_pri[-1]].values
            if len(_row) == len(ok):
                _row = _row[ok]
            if len(_row) == len(corr):
                _same = (_row[:, None] == _row[None, :]) & pd.notna(_row)[:, None]
                corr = np.where(_same, corr, -1.0)'''


def industry_peers(src):
    a = "    peer_insts = daily_ret.columns.values\n"
    assert src.count(a) == 1
    src = src.replace(a, "    peer_insts = daily_ret.columns.values\n" + IND_SQL)
    b = "        corr = np.nan_to_num(corr, nan=-1.0)          # 缺失相关性记 -1（等价于不选）\n"
    assert src.count(b) == 1
    return src.replace(b, b + IND_MASK + "\n")


SPECS = [
    ("因子代码/129_竞价放量附加.py", ("c_auc",), (0.15,), False, None,
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 集合竞价放量附加（新方向，09:30 修复版）。

    第一次使用 **09:25 集合竞价 bar 的成交额**（此前所有因子只用它的价格来补 09:30 截面，从未用它的量）：
      信号 = -(今日竞价成交额 / 过去 5 个交易日竞价成交额均值)，即竞价相对自身历史的放量倍数（只用过去数据）。
    依据：集合竞价是信息与情绪最集中的时段，竞价放量（尤其伴随高开）意味着开盘前的抢筹/情绪定价，
      之后容易回落（开盘竞价的信息含量与后续反转，见 Berkman 等 2012 关于开盘竞价交易的研究）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/130_竞价额占比附加.py", ("c_aucshare",), (0.15,), False, None,
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 竞价额占比附加（新方向，09:30 修复版）。

    与 129 号同源（09:25 集合竞价成交额），但分母换成过去 5 个交易日**全天**成交额均值，
      即「竞价成交额相对该股正常日成交规模的比重」，直接刻画开盘前定价的规模冲击，取负。
    129 与 130 分别评测，判断竞价信号的可用形态是「相对自身竞价的放量」还是「相对日常成交的规模」。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/131_分钟净买压附加.py", ("c_netflow",), (0.15,), False, None,
     """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 分钟净买压附加（新方向，09:30 修复版）。

    窗口内按分钟涨跌方向给成交额赋正负号：(上涨分钟成交额 - 下跌分钟成交额) / 窗口成交额，
      最近 40 个窗口的均值取负。这是「bar 级主动买卖不平衡」的代理：不依赖逐笔方向（那已被否掉），
      只依赖分钟收盘价相对开盘价的方向，数据上完全可得，且与 121 号的「窗口收盘价相对窗口 VWAP 位置」
      不是同一个量（后者与分钟方向无关）。09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    ("因子代码/132_行业内部软同伴.py", (), (), True, None,
     """行业内部软同伴残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    同伴的候选范围从「全市场相关性最高」改为「**同一申万一级行业内部**相关性最高」的 100 只，
      仍按正相关度 2 次方加权（软同伴）。行业码取月初之前最近一个交易日的官方行业暴露（只用过去数据）。
    依据：平台评测会剔除 32 个行业哑变量，因此同伴均值里属于行业的部分会被一起剔掉，同伴的价值主要
      来自行业内部的相对过度反应；把候选限制在同行业，等价于把「伪行业」定义收敛到真实行业内部
      （因子研究资料库第四节的建议）。09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
]

PROBES = [
    ("因子实验/probe_netflow.py", "c_netflow", "分钟净买压（单独成因子，探针）"),
    ("因子实验/probe_auc.py", "c_auc", "集合竞价放量（单独成因子，探针）"),
]

if __name__ == "__main__":
    for path, cols, weights, ind, _u, text in SPECS:
        s = soft_peer(BASE, power=2.0)
        s = patch_sql(s)
        if ind:
            s = s.replace("    instruments_table = datasources.get(\"instruments\", \"cpt_jyc_2026_instruments\")",
                          "    instruments_table = datasources.get(\"instruments\", \"cpt_jyc_2026_instruments\")\n"
                          "    exposure_table = datasources.get(\"exposure\", \"cpt_jyc_2026_exposure\")")
            s = industry_peers(s)
            s = tail(s, (), ())
        else:
            s = add_snips(s, cols)
            s = keep(s, cols)
            s = tail(s, cols, weights)
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
    for path, col, text in PROBES:
        s = soft_peer(BASE, power=2.0)
        s = patch_sql(s)
        s = add_snips(s, (col,))
        s = keep(s, (col,))
        s = probe_tail(s, col)
        s = doc(s, "%s。\n\n    因子值只由该分量经四特征 1.5 倍中性化得到（不含阶梯），用于测量新信号本身的信息含量。\n"
                   "    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。" % text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
