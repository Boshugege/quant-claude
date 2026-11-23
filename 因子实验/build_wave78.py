# -*- coding: utf-8 -*-
"""第七十八批：把「日内低通平滑」这个换手杠杆做两条梯度，并把它用在两个新信息源上（解决它们的换手问题）。

- 525/526：头部 + 日内低通 alpha=0.85 / 0.35（514 是 0.6；0.6 已把换手从 0.567 压到 0.3869、公榜估计 0.8375）。
- 527：503（市值分组条件反转）的低通版 —— 该族单独 IC 好但换手 0.592，低通后有望变成能单独交付的候选。
- 528：520（近档挂单集中度分组）的低通版。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

SMOOTH_HEAD = '''    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(lambda x: x.ewm(alpha=%s, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''

OLD_HEAD = '''    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''

OLD_STAND = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
NEW_STAND = '''    result["signal"] = result["c_condrev"]
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = result["signal"]'''

# 头部低通梯度：486 骨架里没有低通，这里统一用 514 的骨架（=486 + 低通 0.6）作为出发点。
H514 = (CODE / "514_头部加日内信号低通平滑.py").read_text(encoding="utf-8")
assert 'alpha=0.6' in H514
for num, alpha, fname in [("525", "0.85", "525_头部加低通零点八五.py"), ("526", "0.35", "526_头部加低通零点三五.py")]:
    body = H514.replace("alpha=0.6", "alpha=%s" % alpha)
    body = body.replace('"""头部 + 日内信号 EWMA 低通平滑（换手杠杆，头部变体）。',
                        '"""头部 + 日内信号 EWMA 低通平滑（alpha=%s 梯度档，换手杠杆）。' % alpha, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)

for num, base, fname, desc in [
    ("527", "503_单独市值分组条件反转.py", "527_单独市值分组条件反转低通版.py", "市值分组条件反转 + 日内低通平滑（低换手新源版）"),
    ("528", "520_单独近档挂单集中度分组条件反转.py", "528_单独近档挂单集中度分组条件反转低通版.py", "近档挂单集中度分组条件反转 + 日内低通平滑（低换手新源版）"),
]:
    body = (CODE / base).read_text(encoding="utf-8")
    assert OLD_STAND in body, base
    body = body.replace(OLD_STAND, NEW_STAND)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
