# -*- coding: utf-8 -*-
"""第八十四批：低通 alpha 在 0.7 以上继续取点（544=0.8528 已是目前最高），并把配方再套到两个族内轴上。"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

h544 = (CODE / "544_头部加挂单减权低通零点七.py").read_text(encoding="utf-8")
assert "alpha=0.7" in h544
b549 = h544.replace("alpha=0.7", "alpha=0.75")
b549 = b549.replace('"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.7（低通档内细调）。',
                    '"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.75。', 1)
(CODE / "549_头部加挂单减权低通零点七五.py").write_text(b549, encoding="utf-8")
print("写出 549_头部加挂单减权低通零点七五.py")

b550 = h544.replace("_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10])",
                    "_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15])")
b550 = b550.replace('"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.7（低通档内细调）。',
                    '"""头部 + 一至三档挂单不平衡 0.15 + 中性化 2.0 倍 + 日内低通 0.7。', 1)
(CODE / "550_头部加挂单零点一五低通零点七.py").write_text(b550, encoding="utf-8")
print("写出 550_头部加挂单零点一五低通零点七.py")

SMOOTH = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["signal"] = result["c_condrev"]
    _ord = result.sort_values(["instrument", "date"])
    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(
        lambda x: x.ewm(alpha=0.6, adjust=False).mean())
    result["signal"] = _ew.reindex(result.index)
    result["factor"] = neutralize("signal", CHARS4, %s)'''
OLD = '''    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
for base, frac, fname, desc in [
    ("542_单独成交笔数分组条件反转.py", "2.0", "551_单独成交笔数分组低通二点零.py", "成交笔数分组 + 低通 0.6 + 中性化 2.0"),
    ("535_单独盘口报价深度分组条件反转.py", "2.5", "552_单独盘口报价深度分组低通二点五.py", "盘口报价深度分组 + 低通 0.6 + 中性化 2.5"),
]:
    body = (CODE / base).read_text(encoding="utf-8")
    assert OLD in body, base
    body = body.replace(OLD, SMOOTH % frac)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
