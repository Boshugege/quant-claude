# -*- coding: utf-8 -*-
"""第八十一批：把「低通」这一杠杆再做两个方向的加深。

- 537：头部 + 日内低通 0.6 + 跨日低通 0.5（两级低通，进一步压换手）。
- 538：头部 + 日内低通 0.6 + 中性化 2.5 倍（探针：低换手 + 去 104 化）。
- 539/540：把两个盘口新源的分组版做成「低通 + 中性化 2.0」，看能否第一次单独站上 0.80 档。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

h514 = (CODE / "514_头部加日内信号低通平滑.py").read_text(encoding="utf-8")
assert "alpha=0.6" in h514

# 537：在 514 的低通之后再加一层跨日（同一 hhmm 截面）低通
anchor = '''    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)'''
assert anchor in h514
new537 = '''    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    _tmp = _ord.assign(_s=_ew)
    _hh = _tmp["date"].dt.strftime("%H%M")
    _ew2 = _tmp.groupby(["instrument", _hh], sort=False)["_s"].transform(
        lambda x: x.ewm(alpha=0.5, adjust=False).mean())
    result["signal"] = _ew2.reindex(result.index)'''
b537 = h514.replace(anchor, new537)
b537 = b537.replace('"""头部 + 日内信号 EWMA 低通平滑（换手杠杆，头部变体）。',
                    '"""头部 + 日内低通 0.6 + 跨日低通 0.5（两级低通，进一步压换手）。', 1)
(CODE / "537_头部加日内跨日两级低通.py").write_text(b537, encoding="utf-8")
print("写出 537_头部加日内跨日两级低通.py")

# 538：514 的中性化 1.5 → 2.5
b538 = h514.replace('neutralize("signal", CHARS4, 1.5)', 'neutralize("signal", CHARS4, 2.5)')
b538 = b538.replace('"""头部 + 日内信号 EWMA 低通平滑（换手杠杆，头部变体）。',
                    '"""头部 + 日内低通 0.6 + 中性化 2.5 倍（探针：低换手 + 去 104 化，私榜风险高）。', 1)
(CODE / "538_头部加低通与二点五中性化探针.py").write_text(b538, encoding="utf-8")
print("写出 538_头部加低通与二点五中性化探针.py")

# 539/540：新源 + 低通 + 中性化 2.0（去均值族骨架 + 两级杠杆）
SMOOTH = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["signal"] = result["c_condrev"]
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, 2.0)'''
OLD = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
for base, fname, desc in [
    ("503_单独市值分组条件反转.py", "539_单独市值分组条件反转低通二点零.py", "市值分组条件反转 + 低通 0.6 + 中性化 2.0（新源的低换手去相关版）"),
    ("535_单独盘口报价深度分组条件反转.py", "540_单独盘口报价深度分组低通二点零.py", "盘口报价深度分组条件反转 + 低通 0.6 + 中性化 2.0"),
]:
    body = (CODE / base).read_text(encoding="utf-8")
    assert OLD in body, base
    body = body.replace(OLD, SMOOTH)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
