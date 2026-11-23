# -*- coding: utf-8 -*-
"""第十九批（198–204）：日级（日内恒定）分量 × 104 阶梯的权重族。

思路（依据任务说明第 0 节的新校准 + 我们的配对比较结论）：
- 头部区分度来自 IC_IR、压力、换手；104 的短板就是换手（赋分表里 weakest=turnover）。
- 换手按"同一交易日内相邻截面的成分替换比例"统计，所以**日内恒定**的分量对换手零贡献，
  会按权重把日内阶梯带来的churn稀释掉。
- 过去 113–175 试的都是"日**内**微结构小权重附加项"（权重 0.15–0.30），换手 z 全部 +23~+35，
  收益被换手吃掉。本批改成"日**级**分量 + 大权重"，是没试过的一维。
- 日级分量选两个都带反转信息、且日内恒定的：20 日尺度反转（任务说明第 5 节第 2 条）、
  隔夜跳空反转（资料库第 6 条，63 vs 56 号已验证隔夜带反转信息）。
- 09:30 一律输出 0（与 104/103/107 同口径）：平台会跳过全 0 截面，补齐 09:30 会让五项指标全面变差
  （赋分表里 104 为 0.9038、104 补 09:30 只有 0.6305）。204 是"补 09:30"的对照，用来再钉一次这个结论。
"""
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC104 = ROOT / "因子代码/104_同伴残差阶梯过度中性.py"

A_DAY = '    df["r_day"] = safe_log_ratio(df["pre_close"], px)\n'
A_KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max"]\n'
A_COMPOSE = '    result["ladder"] = peer_ladder()\n    result["factor"] = neutralize("ladder", CHARS4, 1.5)\n'
A_FILL = '    result["factor"] = result["factor"].replace([np.inf, -np.inf], np.nan).astype("float64").fillna(0.0)\n'
# 隔夜分量需要的 SQL 补丁
A_SQL_MAXPC = '            MAX(pre_close) AS pre_close\n'
A_SQL_SEL = '        f.sum_amt, f.sum_vol, f.last_px, f.pre_close\n'
A_SQL_DTYPE = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:\n'

DAY_SNIP = '''
    # ---- 日级（日内恒定）分量：反向的 20 日尺度收益，用日收盘价序列构造 ----
    _dpv = df.groupby(["day", "instrument"], sort=False)["last_px"].last().unstack("instrument").sort_index()
    _dpv = _dpv.where(_dpv > 0)
    _d20 = np.log(_dpv.shift(20) / _dpv.shift(1)).stack().rename("d20").reset_index()
    _d20["_k"] = _d20["day"].astype(str) + "|" + _d20["instrument"].astype(str)
    df["d20"] = (df["day"].astype(str) + "|" + df["instrument"].astype(str)).map(_d20.set_index("_k")["d20"])
'''

ON_SNIP = '''
    # ---- 日级（日内恒定）分量：隔夜跳空反转 ln(昨收 / 09:25 集合竞价价) ----
    _don = df.groupby(["day", "instrument"], sort=False).agg(
        _pc=("pre_close", "first"), _aup=("auction_px", "first"))
    _don = np.log(_don["_pc"] / _don["_aup"].where(_don["_aup"] > 0)).unstack("instrument").sort_index()
    _don = _don.stack().rename("don").reset_index()
    _don["_k"] = _don["day"].astype(str) + "|" + _don["instrument"].astype(str)
    df["don"] = (df["day"].astype(str) + "|" + df["instrument"].astype(str)).map(_don.set_index("_k")["don"])
'''

ZERO930 = '''
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面；补齐 09:30 会让五项指标全面变差。
    result.loc[result["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
'''

COMPOSE = '''    result["ladder"] = peer_ladder()
{daycols}    result["signal"] = rank_blend([{blendcols}], [{wts}])
    result["factor"] = neutralize("signal", CHARS4, 1.5)
'''


def build(use20, useon, w20, won, zero930, title, body):
    s = SRC104.read_text(encoding="utf-8")
    assert s.count(A_DAY) == 1 and s.count(A_KEEP) == 1 and s.count(A_COMPOSE) == 1 and s.count(A_FILL) == 1

    snips = ""
    cols, keepcols, wts = [], [], []
    if useon:
        assert s.count(A_SQL_MAXPC) == 1 and s.count(A_SQL_SEL) == 1 and s.count(A_SQL_DTYPE) == 1
        s = s.replace(A_SQL_MAXPC, "            MAX(CASE WHEN hms = '092500' THEN px END) AS auction_px,\n" + A_SQL_MAXPC)
        s = s.replace(A_SQL_SEL, A_SQL_SEL.replace("f.last_px, f.pre_close", "f.last_px, f.pre_close, f.auction_px"))
        s = s.replace(A_SQL_DTYPE, '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close", "auction_px"]:\n')
        s = s.replace('    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)\n',
                      '    df["pre_close"] = df["pre_close"].where(df["pre_close"] > 0)\n'
                      '    df["auction_px"] = df["auction_px"].where(df["auction_px"] > 0)\n')
        snips += ON_SNIP
        cols.append("don")
        keepcols.append("don")
        wts.append(won)
    if use20:
        snips += DAY_SNIP
        cols.append("d20")
        keepcols.append("d20")
        wts.append(w20)

    s = s.replace(A_DAY, A_DAY + snips)
    s = s.replace(A_KEEP, A_KEEP.replace('"c_max"]', '"c_max", ' + ", ".join('"%s"' % c for c in keepcols) + ']'))
    s = s.replace('    df = df.replace([np.inf, -np.inf], np.nan)\n',
                  '    df = df.replace([np.inf, -np.inf], np.nan)\n')
    daycol_lines = "".join('    result["%s"] = peer_free(result, "%s")\n' % (c, c) for c in cols)
    s = s.replace(A_COMPOSE, COMPOSE.format(
        daycols=daycol_lines,
        blendcols=", ".join('"ladder"' if False else '"%s"' % c for c in ["ladder"] + cols),
        wts=", ".join(["1.0"] + ["%g" % w for w in wts]),
    ))
    s = s.replace(A_FILL, A_FILL + (ZERO930 if zero930 else ""))
    # peer_free：把 df 里的日级分量直接搬到 result（不做同伴残差——平台本身会剔除 32 个行业哑变量，
    # 日级分量里的行业共同部分在评分时已经被剔掉）
    helper = '''
    def peer_free(frame, col):
        """把 df 上的日级（日内恒定）分量按 (date, instrument) 搬到 result（缺失记 0，中性）。"""
        m = df[["date", "instrument", col]].drop_duplicates(["date", "instrument"])
        j = frame[["date", "instrument"]].merge(m, on=["date", "instrument"], how="left")
        return pd.Series(j[col].to_numpy(dtype="float64"), index=frame.index)

'''
    anchor = "    def rank_blend(cols, weights=None):"
    assert s.count(anchor) == 1
    s = s.replace(anchor, helper.lstrip("\n") + anchor)
    s = s.replace('    """\n', '    """\n', 1)
    # 替换文档字符串首行说明
    s = s.replace('    """同伴残差阶梯 + 四特征 1.5 倍中性化。',
                  '    """' + title)
    return s, body


SPECS = [
    ("因子代码/198_日级二十日反转二成.py", True, False, 0.25, 0, True,
     "同伴残差阶梯 + 四特征 1.5 倍中性化 + 日级 20 日反转分量（占合成权重 20%）。",
     "日级分量 ln(20 个交易日前的日收盘 / 昨日收盘)，日内恒定。混合权重 [1.0, 0.25] → 日级占 20%。"),
    ("因子代码/199_日级二十日反转对半.py", True, False, 1.0, 0, True,
     "同伴残差阶梯 + 四特征 1.5 倍中性化 + 日级 20 日反转分量（占合成权重 50%）。",
     "同上，权重 [1.0, 1.0] → 日级占 50%。"),
    ("因子代码/200_日级二十日反转七成.py", True, False, 2.0, 0, True,
     "同伴残差阶梯 + 四特征 1.5 倍中性化 + 日级 20 日反转分量（占合成权重 67%）。",
     "同上，权重 [1.0, 2.0] → 日级占 67%。这一档换手应该最低。"),
    ("因子代码/201_日级隔夜反转对半.py", False, True, 0, 1.0, True,
     "同伴残差阶梯 + 四特征 1.5 倍中性化 + 日级隔夜跳空反转分量（占合成权重 50%）。",
     "日级分量 ln(昨收 / 09:25 集合竞价价)，日内恒定（资料库第 6 条）。权重 [1.0, 1.0]。"),
    ("因子代码/202_日级隔夜反转七成.py", False, True, 0, 2.0, True,
     "同伴残差阶梯 + 四特征 1.5 倍中性化 + 日级隔夜跳空反转分量（占合成权重 67%）。",
     "同上，权重 [1.0, 2.0]。"),
    ("因子代码/203_日级双分量各三成.py", True, True, 1.0, 1.0, True,
     "同伴残差阶梯 + 四特征 1.5 倍中性化 + 两个日级分量（20 日反转、隔夜反转，各占 33%）。",
     "两个日级分量各 1.0，合计占 67%。检验两个日级分量是否互补。"),
    ("因子代码/204_日级二十日反转对半补0930.py", True, False, 1.0, 0, False,
     "199 号的对照：日级分量在 09:30 截面照常输出（不置 0）。",
     "其余与 199 完全相同，只把 09:30 截面留成有信号，用来复核「补 09:30 会掉分」。"),
]

if __name__ == "__main__":
    for path, use20, useon, w20, won, zero, title, body in SPECS:
        code, _ = build(use20, useon, w20, won, zero, title, body)
        p = ROOT / path
        assert not p.exists(), "已有同名文件，不允许覆盖：%s" % p
        p.write_text(code, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", path)
