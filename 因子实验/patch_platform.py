# -*- coding: utf-8 -*-
"""从 因子实验/104_09点30修复.py 生成新的平台因子代码（结构性改动：软同伴 / 附加项）。"""
import pathlib
import re

BASE = pathlib.Path("因子实验/104_09点30修复.py").read_text(encoding="utf-8")

PEER_OLD = '''        ok = (hist.notna().sum() >= PEER_MIN_OBS).values
        corr = hist.loc[:, ok].corr(min_periods=PEER_MIN_OBS).values
        np.fill_diagonal(corr, -np.inf)
        corr = np.nan_to_num(corr, nan=-np.inf)
        sub = np.where(ok)[0]
        idx = np.full((len(peer_insts), PEER_K), -1, dtype=np.int64)
        idx[sub] = sub[np.argsort(-corr, axis=1, kind="stable")[:, :PEER_K]]
        peers_by_month[m] = idx'''

PEER_NEW_SOFT = '''        ok = (hist.notna().sum() >= PEER_MIN_OBS).values
        corr = hist.loc[:, ok].corr(min_periods=PEER_MIN_OBS).values
        np.fill_diagonal(corr, np.nan)
        corr = np.nan_to_num(corr, nan=-1.0)          # 缺失相关性记 -1（等价于不选）
        sub = np.where(ok)[0]
        top = np.argsort(-corr, axis=1, kind="stable")[:, :PEER_K]
        idx = np.full((len(peer_insts), PEER_K), -1, dtype=np.int64)
        idx[sub] = sub[top]
        cvals = corr[np.arange(len(corr))[:, None], top]
        wts = np.clip(cvals, 0.0, None) ** PEER_POWER   # 软同伴：正相关度加权
        wsum = wts.sum(axis=1, keepdims=True)
        wts = np.where(wsum > 0, wts / np.where(wsum > 0, wsum, 1.0), 0.0)
        full_w = np.zeros((len(peer_insts), PEER_K))
        full_w[sub] = wts
        peers_by_month[m] = (idx, full_w)'''

PEER_MEAN_OLD = '''            idx = peers_by_month.get(m)
            if idx is None:
                continue
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            valid = idx >= 0
            g = wide.values[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            cnt = (~np.isnan(g)).sum(axis=2)
            pm = np.where(cnt >= 3, np.nansum(g, axis=2) / np.maximum(cnt, 1), np.nan)'''

PEER_MEAN_NEW = '''            rec = peers_by_month.get(m)
            if rec is None:
                continue
            idx, wts = rec
            sub = result.iloc[rows]
            wide = sub.pivot(index="date", columns="instrument", values=col).reindex(columns=peer_insts)
            valid = idx >= 0
            g = wide.values[:, np.where(valid, idx, 0)]
            g = np.where(valid[None, :, :], g, np.nan)
            ww = np.where(valid[None, :, :], wts[None, :, :], 0.0)
            okm = ~np.isnan(g)
            cnt = okm.sum(axis=2)
            wsum = (ww * okm).sum(axis=2)
            pm = np.where((cnt >= 3) & (wsum > 0), np.nansum(np.where(okm, g, 0.0) * ww, axis=2) / np.where(wsum > 0, wsum, 1.0), np.nan)'''

CONST_OLD = "    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS = 30, 60, 30"
CONST_SOFT = "    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 100, 60, 30, 2.0"

KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max"]'
TAIL_OLD = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["factor"] = neutralize("ladder", CHARS4, 1.5)'''

SQL_WIN_OLD = '''            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(pre_close) AS pre_close'''
SQL_WIN_NEW = '''            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,
            MAX(CASE WHEN px_high > 0 THEN px_high END) AS win_high,
            MIN(CASE WHEN px_low > 0 THEN px_low END) AS win_low,
            MAX(pre_close) AS pre_close'''

SNAPSHOT_OLD = """            trading_day,
            hms,
            pre_close,"""
SNAPSHOT_NEW = """            trading_day,
            hms,
            pre_close,
            px_high,
            px_low,"""

SEL_OLD = "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close"
SEL_NEW = "        f.sum_amt, f.sum_vol, f.last_px, f.win_high, f.win_low, f.pre_close"

FLT_OLD = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:'
FLT_NEW = '    for c in ["sum_amt", "sum_vol", "last_px", "win_high", "win_low", "pre_close"]:'


def add_dsd(src):
    s = '''    df["c_max"] = -roll(win_ret, 40, "max", 10)\n'''
    assert src.count(s) == 1
    return src.replace(s, s + '''
    # 下行波动占比：最近 40 个窗口负收益平方和 / 全部收益平方和。
    df["c_dsd"] = roll(win_ret.where(win_ret < 0, 0.0) ** 2, 40, "sum", 8) / roll(win_ret ** 2, 40, "sum", 8).replace(0.0, np.nan)
''')


def add_amp(src):
    assert src.count(SNAPSHOT_OLD) == 1
    src = src.replace(SNAPSHOT_OLD, SNAPSHOT_NEW)
    assert src.count(SQL_WIN_OLD) == 1
    out = src.replace(SQL_WIN_OLD, SQL_WIN_NEW)
    assert out.count(SEL_OLD) == 1
    out = out.replace(SEL_OLD, SEL_NEW)
    assert out.count(FLT_OLD) == 1
    out = out.replace(FLT_OLD, FLT_NEW)
    anchor = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'
    assert out.count(anchor) == 1
    out = out.replace(anchor, anchor + '''    # 振幅（取负）：最近 20 个窗口 (最高 - 最低) / 昨收 的均值。
    _amp = (df["win_high"] - df["win_low"]) / df["pre_close"].where(df["pre_close"] > 0)
    df["c_amp"] = -roll(_amp, 20, "mean", 5)
''')
    return out


def soft_peer(src, power=2.0):
    assert src.count(CONST_OLD) == 1
    out = src.replace(CONST_OLD, CONST_SOFT.replace("2.0", "%.1f" % power))
    assert out.count(PEER_OLD) == 1
    out = out.replace(PEER_OLD, PEER_NEW_SOFT)
    assert out.count(PEER_MEAN_OLD) == 1
    return out.replace(PEER_MEAN_OLD, PEER_MEAN_NEW)


def tail(src, extra_cols=(), weights=()):
    cols = ["ladder"] + list(extra_cols)
    if not extra_cols:
        return src.replace(TAIL_OLD, TAIL_OLD.replace('neutralize("ladder"', 'neutralize("ladder"'))
    ws = [1.0] + list(weights)
    new = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["signal"] = rank_blend(%s, %s)
    result["factor"] = neutralize("signal", CHARS4, 1.5)''' % (repr(cols).replace("'", '"'), repr(ws).replace("'", '"'))
    assert src.count(TAIL_OLD) == 1
    return src.replace(TAIL_OLD, new)


def keep(src, extra):
    assert src.count(KEEP_OLD) == 1
    cols = '"r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max"'
    # 自动补：阶梯里用到的尺度列必须进 keep，否则合并后 result 里没有该列（164/173/178/181 曾因此整批 KeyError）。
    auto = [c for c in ("r_rv16", "r_bv4", "r_bv8") if 'df["%s"]' % c in src and '"%s"' % c not in cols]
    names = list(dict.fromkeys(list(extra) + auto))
    return src.replace(KEEP_OLD, '    keep = [%s, %s]' % (cols, ", ".join('"%s"' % c for c in names)))


def doc(src, text):
    old = src.split('"""')[1]
    return src.replace('"""%s"""' % old, '"""%s"""' % text, 1)



def add_rv16(src):
    a = '    df["r_rv8"] = rolling_vwap_rev(8)\n'
    assert src.count(a) == 1
    src = src.replace(a, a + '    df["r_rv16"] = rolling_vwap_rev(16)\n')
    b = '        for c in ["r_30", "r_rv4", "r_rv8", "r_day"]:'
    assert src.count(b) == 1
    return src.replace(b, '        for c in ["r_30", "r_rv4", "r_rv8", "r_rv16", "r_day"]:')


FIX_OLD = """            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment
                      OR (time_segment = 93000 AND hms = '092500') THEN px ELSE 0 END) AS last_px,"""
FIX_NEW = """            MAX(CASE WHEN CAST(hms AS INTEGER) = time_segment THEN px ELSE 0 END) AS last_px,"""


def nofix(src):
    assert src.count(FIX_OLD) == 1
    return src.replace(FIX_OLD, FIX_NEW)

SPECS = {
    "因子代码/114_软同伴残差阶梯四特征中性.py": dict(
        soft=True, soft_power=None, extras=(), weights=(), keep_extra=(), rv16=False,
        doc="""软同伴残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    与 104 号的区别只在同伴的定义：104 号硬选相关性最高的 30 只同伴等权平均，这里改成
      “软同伴”——取相关性最高的 100 只，按正相关度的 2 次方加权（归一），即相关度越高权重越大，
      相关度为负的同伴权重置 0。依据：行业内反转强于全市场反转（Da, Liu & Schaumburg 2014），
      但“行业”是硬边界，用收益相关性做连续加权的伪行业更贴近真实的信息传导强度（因子研究资料库第四节）。
    09:30 修复同 104 号对照版本。"""),
    "因子代码/115_同伴残差阶梯振幅附加.py": dict(
        soft=False, soft_power=None, extras=("c_amp",), weights=(0.20,), keep_extra=("c_amp",), rv16=False,
        doc="""同伴残差阶梯 + 四特征 1.5 倍中性化 + 振幅附加（新方向，09:30 修复版）。

    在 104 号基础上加 0.20 权重的“低振幅”附加项（最近 20 个窗口 (最高-最低)/昨收 的均值取负）。
    依据：低波动/低振幅异象（Ang 等 2006 特质波动之谜在 A 股的体现），且振幅是官方风格里没有直接
      覆盖的特征；本地筛选（2023 验证集）显示叠加后 IC_IR 1.619→1.700、压力 1.541→1.630 明显提升。
    09:30 修复同 104 号对照版本。"""),
    "因子代码/116_软同伴振幅下行附加.py": dict(
        soft=True, soft_power=None, extras=("c_amp", "c_dsd"), weights=(0.15, 0.15), keep_extra=("c_amp", "c_dsd"), rv16=False,
        doc="""软同伴残差阶梯 + 四特征 1.5 倍中性化 + 振幅与下行波动附加（新方向，09:30 修复版）。

    同伴改用软同伴（相关性加权），并叠加 0.15 权重的低振幅项与 0.15 权重的下行波动占比项。
    本地筛选（2023 验证集）中该组合的 IC/IC_IR/压力均为本轮最高（0.0814 / 1.687 / 1.610）。
    09:30 修复同 104 号对照版本。"""),
    "因子代码/117_软同伴一倍相关加权.py": dict(
        soft_power=1.0, extras=(), weights=(), keep_extra=(), rv16=False,
        doc="""软同伴残差阶梯 + 四特征 1.5 倍中性化（一倍相关度加权，09:30 修复版）。

    与 114 号同结构，但软同伴权重用正相关度的 1 次方（114 号用 2 次方），用于确认加权强度。
    09:30 修复同 104 号对照版本。"""),
    "因子代码/118_软同伴四特征二倍中性.py": dict(
        soft_power=2.0, extras=(), weights=(), keep_extra=(), rv16=False, frac=2.0,
        doc="""软同伴残差阶梯 + 四特征 2.0 倍中性化（09:30 修复版）。

    同伴用软同伴（相关性加权），中性化力度取 2.0 倍（104 号是 1.5 倍）。用于确认“同伴定义 + 中性化力度”
    两个杠杆能否叠加。09:30 修复同 104 号对照版本。"""),
    "因子代码/119_六尺度同伴残差阶梯.py": dict(
        soft_power=2.0, extras=(), weights=(), keep_extra=("r_rv16",), rv16=True,
        doc="""六尺度同伴残差阶梯 + 四特征 1.5 倍中性化（新方向，09:30 修复版）。

    在同伴残差阶梯里增加约 2 个交易日（16 个 30 分钟窗口）的滚动 VWAP 偏离分量，尺度谱从
    {30 分钟, 2 小时, 4 小时, 昨收} 扩到 {…, 约 2 日}。依据：端到端多尺度专家增加长尺度后
    IC_IR、压力稳定性双升；本地粗筛显示加入该分量后压力 1.322→1.343、换手 0.629→0.580。
    同伴用软同伴（相关性加权）。09:30 修复同 104 号对照版本。"""),
    "因子代码/120_软同伴残差不补0930对照.py": dict(
        soft_power=2.0, extras=(), weights=(), keep_extra=(), rv16=False, nofix=True,
        doc="""软同伴残差阶梯 + 四特征 1.5 倍中性化（**不补 09:30 截面**，平台原缺陷版）。

    与 114 号完全相同，只把窗口末价条件还原成平台原始的写法（09:30 窗口取不到末价 → 该截面因子恒为 0）。
    用途：与 114 号构成“补 / 不补 09:30”的对照提交，用真实公榜分数判定平台对常数截面的处理方式，
      从而确认后续所有因子应该采用哪一版。"""),
}

if __name__ == "__main__":
    for path, sp in SPECS.items():
        s = BASE
        _power = sp.get("soft_power")
        if sp.get("soft") and _power is None:
            s = soft_peer(s)
        elif _power is not None:
            s = soft_peer(s, power=_power)
        if sp.get("rv16"):
            s = add_rv16(s)
        if sp.get("nofix"):
            s = nofix(s)
        if "c_dsd" in sp["keep_extra"]:
            s = add_dsd(s)
        if "c_amp" in sp["keep_extra"]:
            s = add_amp(s)
        if sp["keep_extra"]:
            s = keep(s, sp["keep_extra"])
        s = tail(s, sp["extras"], sp["weights"])
        if sp.get("frac"):
            s = s.replace('neutralize("signal", CHARS4, 1.5)', 'neutralize("signal", CHARS4, %.1f)' % sp["frac"])
            s = s.replace('neutralize("ladder", CHARS4, 1.5)', 'neutralize("ladder", CHARS4, %.1f)' % sp["frac"])
        s = doc(s, sp["doc"])
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
