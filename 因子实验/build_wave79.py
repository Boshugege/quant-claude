# -*- coding: utf-8 -*-
"""第七十九批：低通杠杆的组合与推广。

- 529：514（头部 + 低通 0.6）+ 中性化 2.0 倍（把公榜最高的换手档与已证实的中性化杠杆叠起来）。
- 530：510（头部 + 市值分组条件反转 0.15）+ 低通 0.6（把 IC 最好的叠加项与换手杠杆叠起来）。
- 531：522（行业内部盘口不平衡残差分组）的通低版。
- 532：521（盘口不平衡日内变化分组）的通低版。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

SMOOTH_TAIL = '''    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)'''

# 529：514 的低通块里把中性化 1.5 换成 2.0
h514 = (CODE / "514_头部加日内信号低通平滑.py").read_text(encoding="utf-8")
assert 'alpha=0.6' in h514 and 'neutralize("signal", CHARS4, 1.5)' in h514
b529 = h514.replace('neutralize("signal", CHARS4, 1.5)', 'neutralize("signal", CHARS4, 2.0)')
b529 = b529.replace('"""头部 + 日内信号 EWMA 低通平滑（换手杠杆，头部变体）。',
                    '"""头部 + 日内信号 EWMA 低通平滑（alpha=0.6）+ 中性化 2.0 倍（两个最有效杠杆叠加）。', 1)
(CODE / "529_头部加低通与二点零中性化.py").write_text(b529, encoding="utf-8")
print("写出 529_头部加低通与二点零中性化.py")

# 530：510（头部 + c_condrev 0.15）的合成块前插入低通
h510 = (CODE / "510_头部加市值分组条件反转.py").read_text(encoding="utf-8")
anchor510 = '''    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
assert anchor510 in h510
new510 = '''    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)
''' + SMOOTH_TAIL + '''
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
b530 = h510.replace(anchor510, new510)
b530 = b530.replace('"""头部 + 市值分组条件反转 0.15。', '"""头部 + 市值分组条件反转 0.15 + 日内信号低通（alpha=0.6）。', 1)
(CODE / "530_头部加市值分组条件反转与低通.py").write_text(b530, encoding="utf-8")
print("写出 530_头部加市值分组条件反转与低通.py")

OLD_STAND = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
NEW_STAND = '''    result["signal"] = result["c_condrev"]
''' + SMOOTH_TAIL + '''
    result["factor"] = result["signal"]'''
for base, fname, desc in [
    ("522_单独行业内部盘口不平衡残差分组条件反转.py", "531_单独行业内部盘口不平衡残差分组低通版.py", "行业内部盘口不平衡残差分组 + 日内低通（低换手新源版）"),
    ("521_单独盘口不平衡日内变化分组条件反转.py", "532_单独盘口不平衡日内变化分组低通版.py", "盘口不平衡日内变化分组 + 日内低通（低换手新源版）"),
]:
    body = (CODE / base).read_text(encoding="utf-8")
    assert OLD_STAND in body, base
    body = body.replace(OLD_STAND, NEW_STAND)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
