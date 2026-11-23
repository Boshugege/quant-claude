# -*- coding: utf-8 -*-
"""第一百零二批：翻转已实测反号的两个平滑合成（613/614）+ 头部叠加 span5 平滑新源（611/612）+ 两个新同伴定义（615/616）。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

def rd(n): return (CODE / n).read_text(encoding="utf-8")
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)

# ---- 611/612：合法头部 + 595（span5 平滑同伴）叠加 ----
s589 = rd("589_头部加平滑同伴残差015.py")
s611 = s589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1)
s611 = s611.replace("合法头部 + 567 新源的 10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。",
                    "合法头部 + 567 新源的 5 窗口平滑版 0.15（与 104 相关约 0.62 的新源，单独 IC_IR 1.03）。", 1)
wr("611_头部加平滑同伴残差span5015.py", s611)

s612 = s589.replace("span=10, min_periods=5", "span=5, min_periods=3", 1)
s612 = s612.replace("[1.0, 0.30, 0.15, 0.20, 0.15, 0.15]", "[1.0, 0.30, 0.15, 0.20, 0.15, 0.30]", 1)
s612 = s612.replace("合法头部 + 567 新源的 10 窗口平滑版 0.15（低换手、与 104 相关约 0.50 的新源）。",
                    "合法头部 + 567 新源的 5 窗口平滑版 0.30（与 104 相关约 0.62 的新源，单独 IC_IR 1.03）。", 1)
wr("612_头部加平滑同伴残差span5030.py", s612)

# ---- 613/614：翻转已实测反号的平滑合成 ----
for src, dst, desc in [
    ("593_单独多同伴定义合成平滑版.py", "613_单独多同伴定义合成平滑版翻转.py",
     "多同伴定义合成平滑版（方向已校正，实测反号后取负）"),
    ("596_单独市值与小市值同伴合成平滑版.py", "614_单独市值与小市值同伴合成平滑版翻转.py",
     "市值加权与小市值加权同伴合成平滑版（方向已校正，实测反号后取负）"),
]:
    t = rd(src)
    old = '    result["factor"] = result["signal"]\n'
    assert old in t, src
    t = t.replace(old, old + '    result["factor"] = -result["factor"]  # 实测 IC 为负，取反\n', 1)
    # 尽量更新首行说明（可能存在，不存在则忽略）
    lines = t.split("\n")
    for i, ln in enumerate(lines):
        if ln.strip().startswith('"""') and i > 0 and lines[i].strip() == '"""' and i <= 3:
            pass
    wr(dst, t)

# ---- 615/616：新同伴定义（BETA / BTOP 五分位） ----
S561 = rd("561_单独行业内市值加权同伴残差反转.py")
OLD = '''    # ===== 新轴 561：行业内「前一日市值加权同伴」残差反转（同伴定义 = 官方申万一级行业内，权重 = 前一日流通市值） =====
    _mkw = df["mcap"].fillna(0.0)
    _pk66 = [df["date"].values, df["ind_code"].values]'''
OLD_DOC = "行业内前一日市值加权同伴残差反转（申万一级行业内、前一日流通市值加权；全部暴露表列按交易日错一天）"
assert OLD in S561 and OLD_DOC in S561
for fname, block, desc in [
    ("615_单独行业内BETA五分位同伴残差反转.py",
     '''    # ===== 新轴 615：行业内「前一日 BETA 五分位」同伴残差反转（同伴 = 同行业且前一日 BETA 暴露同五分位，等权） =====
    _bv615 = df["s_BETA"]
    _q615 = ((_bv615.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _q615.values]''',
     "行业内前一日 BETA 五分位同伴残差反转（同伴 = 申万一级行业 × 前一日 BETA 暴露五分位，等权；暴露表错一天）"),
    ("616_单独行业内BTOP五分位同伴残差反转.py",
     '''    # ===== 新轴 616：行业内「前一日 BTOP 五分位」同伴残差反转（同伴 = 同行业且前一日 BTOP 暴露同五分位，等权） =====
    _tv616 = df["s_BTOP"]
    _q616 = ((_tv616.groupby([df["date"], df["ind_code"]]).rank(pct=True) * 5).clip(upper=4.999).fillna(-1)).astype(int)
    _mkw = pd.Series(1.0, index=df.index)
    _pk66 = [df["date"].values, df["ind_code"].values, _q616.values]''',
     "行业内前一日 BTOP 五分位同伴残差反转（同伴 = 申万一级行业 × 前一日 BTOP 暴露五分位，等权；暴露表错一天）"),
]:
    b = S561.replace(OLD, block, 1).replace(OLD_DOC, desc, 1)
    wr(fname, b)
