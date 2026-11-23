# -*- coding: utf-8 -*-
"""第二十批（205–210）：把"换手"这一唯一有余量的方向拆开做单因素对照。

依据（2026-10-11 的配对比较）：104 的赋分表短板是换手（raw 0.605，池内百分位 0.748），
其余四项已经 0.92–0.995，几乎没有余量；而过去所有"104 + 日内小权重附加项"的候选换手 z 都是 +23~+35。
所以本批只做能直接改变日内 churn 的三件事，每个都留对照：
  1) 阶梯内部权重偏向长窗口（r_30 : r_rv4 : r_rv8 : r_day = 0.5 : 1 : 1.5 : 2）——短窗分量是 churn 的主要来源；205 单独测它。
  2) 日级（日内恒定）分量，权重 50%~80%，并测同伴残差版（206/209/210/208）。
  3) 相邻截面 rank 各半（时间平滑），在**新口径**下再验一次"得不偿失"的旧结论（207）。
09:30 一律输出 0（与 104/103/107 同口径；平台跳过全 0 截面）。
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
A_LADDER_RET = '        return rank_blend(cols)\n'
HELPER_ANCHOR = "    def rank_blend(cols, weights=None):"

DAY_SNIP = '''
    # ---- 日级（日内恒定）分量：反向的 20 日尺度收益，用日收盘价序列构造 ----
    _dpv = df.groupby(["day", "instrument"], sort=False)["last_px"].last().unstack("instrument").sort_index()
    _dpv = _dpv.where(_dpv > 0)
    _d20 = np.log(_dpv.shift(20) / _dpv.shift(1)).stack().rename("d20").reset_index()
    _d20["_k"] = _d20["day"].astype(str) + "|" + _d20["instrument"].astype(str)
    df["d20"] = (df["day"].astype(str) + "|" + df["instrument"].astype(str)).map(_d20.set_index("_k")["d20"])
'''

SMOOTH = '''
    # 相邻截面 rank 各半（同一交易日内）——显式复核"时间平滑降换手"的旧结论。
    _rk = result.groupby("date")["factor"].rank(pct=True)
    _tmp = result.assign(_rk=_rk).sort_values(["instrument", "date"])
    _prev = _tmp.groupby(["instrument", _tmp["date"].dt.normalize()], sort=False)["_rk"].shift(1)
    _sm = (0.5 * _tmp["_rk"] + 0.5 * _prev.fillna(_tmp["_rk"])).reindex(result.index)
    result["factor"] = _sm.to_numpy(dtype="float64")
'''

ZERO930 = '''
    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面；补齐 09:30 会让五项指标全面变差。
    result.loc[result["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0
'''

HELPER = '''
    def peer_free(frame, col, peer_index=False):
        """把 df 上的日级（日内恒定）分量搬到 result。peer_index=True 时再减同伴均值（与阶梯分量同口径）。"""
        m = df[["date", "instrument", col]].drop_duplicates(["date", "instrument"])
        j = frame[["date", "instrument"]].merge(m, on=["date", "instrument"], how="left")
        v = pd.Series(j[col].to_numpy(dtype="float64"), index=frame.index)
        frame["_t"] = v
        if peer_index:
            v = v - peer_mean("_t").fillna(frame.groupby("date")["_t"].transform("mean"))
        frame.drop(columns=["_t"], inplace=True)
        return v

'''


def build(title, skew, day_w, peer_index, smooth):
    s = SRC104.read_text(encoding="utf-8")
    assert s.count(A_DAY) == 1 and s.count(A_KEEP) == 1 and s.count(A_COMPOSE) == 1 and s.count(A_FILL) == 1
    if skew:
        assert s.count(A_LADDER_RET) == 1
        s = s.replace(A_LADDER_RET, '        return rank_blend(cols, [0.5, 1.0, 1.5, 2.0])\n')

    day_lines = ""
    if day_w:
        s = s.replace(A_DAY, A_DAY + DAY_SNIP)
        s = s.replace(A_KEEP, A_KEEP.replace('"c_max"]', '"c_max", "d20"]'))
        day_lines = '    result["d20"] = peer_free(result, "d20", peer_index=%s)\n' % (
            "True" if peer_index else "False")
        blend = 'rank_blend(["ladder", "d20"], [1.0, %g])' % day_w
    else:
        blend = '"ladder"'
    s = s.replace(A_COMPOSE, '    result["ladder"] = peer_ladder()\n' + day_lines +
                  '    result["factor"] = neutralize(%s, CHARS4, 1.5)\n' % blend)
    tailcode = ""
    if smooth:
        tailcode += SMOOTH
    tailcode += ZERO930
    s = s.replace(A_FILL, A_FILL + tailcode)
    assert s.count(HELPER_ANCHOR) == 1
    s = s.replace(HELPER_ANCHOR, HELPER.lstrip("\n") + HELPER_ANCHOR)
    s = s.replace('    """同伴残差阶梯 + 四特征 1.5 倍中性化。', '    """' + title)
    return s


SPECS = [
    ("因子代码/205_长窗偏重阶梯.py", "同伴残差阶梯（权重偏向长窗口 r_30:r_rv4:r_rv8:r_day = 0.5:1:1.5:2）+ 四特征 1.5 倍中性化。",
     True, 0, False, False),
    ("因子代码/206_长窗偏重加日级对半.py", "长窗偏重阶梯 + 日级 20 日反转分量（占合成权重 50%）+ 四特征 1.5 倍中性化。",
     True, 1.0, False, False),
    ("因子代码/207_日级对半两截面平滑.py", "日级 20 日反转分量（占 50%）+ 阶梯，合成后再做相邻截面 rank 各半（时间平滑对照）。",
     False, 1.0, False, True),
    ("因子代码/208_日级对半同伴残差.py", "日级 20 日反转分量（占 50%）+ 阶梯，日级分量同样做同伴残差（与阶梯分量同口径）。",
     False, 1.0, True, False),
    ("因子代码/209_日级八成.py", "同伴残差阶梯 + 日级 20 日反转分量（占合成权重 80%，探边界）。",
     False, 4.0, False, False),
    ("因子代码/210_长窗偏重加日级七成.py", "长窗偏重阶梯 + 日级 20 日反转分量（占合成权重 67%）。",
     True, 2.0, False, False),
]

if __name__ == "__main__":
    for path, title, skew, day_w, peer_index, smooth in SPECS:
        code = build(title, skew, day_w, peer_index, smooth)
        p = ROOT / path
        assert not p.exists(), "已有同名文件：%s" % p
        p.write_text(code, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", path)
