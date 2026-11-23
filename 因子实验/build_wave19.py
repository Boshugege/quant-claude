# -*- coding: utf-8 -*-
"""第二十四批（229+）：在 176 骨架（长窗软同伴阶梯 + 分钟上影线，09:30 置零，复算 +0.065/z=3.06）上做增量。

分解结论（待 213–216 回填）显示：176 的收益几乎全来自 IC（+0.176 百分位），换手只有 +0.025；
181 的收益来自换手（+0.101）但 IC 只有 +0.114。本批要找"IC 不掉、换手也降"的组合：
日级（日内恒定）分量对换手零贡献，按权重稀释日内 churn；以及中性化力度、附加项权重。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "因子代码/226_长窗口同伴上影线0930置零.py"

SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.15])\n'
FRAC = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh"]\n'
ANCHOR_RDAY = '    df["r_day"] = safe_log_ratio(df["pre_close"], px)\n'

def day_snip(k, name):
    return ANCHOR_RDAY + '''
    # ---- 日级（日内恒定）分量：反向的 %d 日尺度收益（日收盘价序列）----
    _dpv = df.groupby(["day", "instrument"], sort=False)["last_px"].last().unstack("instrument").sort_index()
    _dpv = _dpv.where(_dpv > 0)
    _d = np.log(_dpv.shift(%d) / _dpv.shift(1)).stack().rename("%s").reset_index()
    _d["_k"] = _d["day"].astype(str) + "|" + _d["instrument"].astype(str)
    df["%s"] = (df["day"].astype(str) + "|" + df["instrument"].astype(str)).map(_d.set_index("_k")["%s"])
''' % (k, k, name, name, name)

PEER_FREE = '''
    def peer_free(frame, col):
        """把 df 上的日级（日内恒定）分量按 (date, instrument) 搬到 result。"""
        m = df[["date", "instrument", col]].drop_duplicates(["date", "instrument"])
        j = frame[["date", "instrument"]].merge(m, on=["date", "instrument"], how="left")
        return pd.Series(j[col].to_numpy(dtype="float64"), index=frame.index)

'''
ANCHOR_RANK = "    def rank_blend(cols, weights=None):"
ANCHOR_MERGE = '    result["ladder"] = peer_ladder()\n    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.15])\n'

SPECS = [
    ("229", "长上影线加日级二十日反转", dict(k=20, col="d20", w=0.2), None, None,
     "叠加日级 20 日反转分量 0.2（日内恒定，稀释 churn）。"),
    ("230", "长上影线加日级五日反转", dict(k=5, col="d5", w=0.2), None, None,
     "叠加日级 5 日反转分量 0.2。"),
    ("231", "长上影线日级对半", dict(k=20, col="d20", w=0.6), None, None,
     "日级 20 日反转分量加重到 0.6。"),
    ("232", "长上影线中性二倍", None, (FRAC, '    result["factor"] = neutralize("signal", CHARS4, 2.0)\n'), None,
     "中性化力度 1.5 → 2.0。"),
    ("233", "长上影线附加三成", None, (SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n'), None,
     "上影线权重 0.15 → 0.30。"),
]

def main():
    s0 = BASE.read_text(encoding="utf-8")
    for num, name, day, sig_repl, _x, note in SPECS:
        s = s0
        if day:
            assert s.count(ANCHOR_RDAY) == 1
            s = s.replace(ANCHOR_RDAY, day_snip(day["k"], day["col"]), 1)
            s = s.replace(KEEP, KEEP[:-2] + ', "%s"]\n' % day["col"], 1)
            assert s.count(ANCHOR_RANK) == 1
            s = s.replace(ANCHOR_RANK, PEER_FREE.lstrip("\n") + ANCHOR_RANK, 1)
            new = ('    result["ladder"] = peer_ladder()\n'
                   '    result["%s"] = peer_free(result, "%s")\n'
                   '    result["signal"] = rank_blend(["ladder", "c_upsh", "%s"], [1.0, 0.15, %s])\n'
                   % (day["col"], day["col"], day["col"], day["w"]))
            assert s.count(ANCHOR_MERGE) == 1
            s = s.replace(ANCHOR_MERGE, new, 1)
        else:
            old, new = sig_repl
            assert s.count(old) == 1, (num, old)
            s = s.replace(old, new, 1)
        s = s.replace('（09:30 置零版）', '（09:30 置零版，%s）' % note, 1)
        p = ROOT / ("因子代码/%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        print("写出", p)
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)

if __name__ == "__main__":
    main()
