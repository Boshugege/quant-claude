# -*- coding: utf-8 -*-
"""第三十四批（278+）：同伴的**规模加权**——任务说明第 5.3 节建议、尚未在平台口径下试过的新方向。

278/279 = 270 骨架，但软同伴权重从"相关度^p"改成"相关度^p × 该同伴历史平均成交额"，即大票的共同运动信息权重更高。
两个版本只差相关度幂次（3.0 / 2.0），用来看规模加权本身是否成立，而不是又去调参。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"
BASE = D / "270_长窗口六特征中性上影线加平均单笔成交额与二日尺度.py"

def repl(t, o, n, tag):
    assert t.count(o) == 1, (tag, o[:70], t.count(o))
    return t.replace(o, n, 1)

s = BASE.read_text(encoding="utf-8")
s = repl(s, '    peer_insts = daily_ret.columns.values\n',
            '    peer_insts = daily_ret.columns.values\n'
            '    daily_amt = df.groupby(["day", "instrument"], sort=False)["sum_amt"].sum().unstack("instrument").reindex(columns=peer_insts).sort_index()\n', "amtmat")
s = repl(s, '        wts = np.clip(cvals, 0.0, None) ** PEER_POWER   # 软同伴：正相关度加权\n',
            '        # 规模加权：相关度^p × 该同伴前 PEER_LOOKBACK 日的平均成交额（大票的共同运动信息权重更高）。\n'
            '        _aw = daily_amt.loc[daily_amt.index < in_month[0]].tail(PEER_LOOKBACK).mean().fillna(0.0).values\n'
            '        _aw = np.where(_aw > 0, _aw, 0.0)\n'
            '        wts = (np.clip(cvals, 0.0, None) ** PEER_POWER) * _aw[sub][:, None]\n', "wts")
for num, name, pw in (("278", "长窗口规模加权同伴加平均单笔与二日尺度", None),
                      ("279", "长窗口规模加权同伴幂二加平均单笔与二日尺度", 2.0)):
    t = s
    if pw is not None:
        t = repl(t, "    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 240, 120, 3.0\n",
                 "    PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 200, 240, 120, 2.0\n", num)
    p = D / ("%s_%s.py" % (num, name))
    p.write_text(t, encoding="utf-8")
    subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
    print("写出", p)
