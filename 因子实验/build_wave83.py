# -*- coding: utf-8 -*-
"""第八十三批：<head>把「低通 + 强中性化」的组合再往前推两点；<new>把配方套到两个 IC 最高的族内轴上。"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

h538 = (CODE / "538_头部加低通与二点五中性化探针.py").read_text(encoding="utf-8")
assert 'neutralize("signal", CHARS4, 2.5)' in h538
b545 = h538.replace('neutralize("signal", CHARS4, 2.5)', 'neutralize("signal", CHARS4, 3.0)')
b545 = b545.replace('"""头部 + 日内低通 0.6 + 中性化 2.5 倍（探针：低换手 + 去 104 化，私榜风险高）。',
                    '"""头部 + 日内低通 0.6 + 中性化 3.0 倍（探针：低换手 + 去 104 化，私榜风险更高）。', 1)
(CODE / "545_头部加低通与三点零中性化探针.py").write_text(b545, encoding="utf-8")
print("写出 545_头部加低通与三点零中性化探针.py")

h534 = (CODE / "534_头部加挂单减权低通二点零中性化.py").read_text(encoding="utf-8")
assert "_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10])" in h534
b546 = h534.replace("_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10])",
                    "_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.05])")
b546 = b546.replace("_W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.07])",
                    "_W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.035])")
b546 = b546.replace('neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 2.5)')
b546 = b546.replace('"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.6（三杠杆的减权版）。',
                    '"""头部 + 一至三档挂单不平衡 0.05 + 中性化 2.5 倍 + 日内低通 0.6。', 1)
(CODE / "546_头部加挂单零点零五低通二点五.py").write_text(b546, encoding="utf-8")
print("写出 546_头部加挂单零点零五低通二点五.py")

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
    ("536_单独盘口不平衡波动分组条件反转.py", "547_单独盘口不平衡波动分组低通二点零.py", "盘口不平衡波动分组 + 低通 0.6 + 中性化 2.0"),
    ("541_单独平均单笔成交额分组条件反转.py", "548_单独平均单笔成交额分组低通二点零.py", "平均单笔成交额分组 + 低通 0.6 + 中性化 2.0"),
]:
    body = (CODE / base).read_text(encoding="utf-8")
    assert OLD in body, base
    body = body.replace(OLD, SMOOTH)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
