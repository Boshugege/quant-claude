# -*- coding: utf-8 -*-
"""第一百四十一批：补满队列。
- 799 / 800：内部得分第一的 433（上影线时点重心族）加 2.0 / 2.5 倍中性化（确认实验）。
- 801：788 的时点重心换成 span5 平滑版（头部 + 新源的低换手版）。
- 802：766 同时加时点重心 0.15 与真换手 0.15 两个新源。
"""
import pathlib, re

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"


def repl(t, old, new, tag):
    assert t.count(old) == 1, (tag, old[:80], t.count(old))
    return t.replace(old, new, 1)


# 799/800：433 中性化
s433 = (D / "433_上影线时点重心.py").read_text(encoding="utf-8")
A433 = '    result["factor"] = neutralize("signal", CHARS4, 1.5)\n'
for num, frac in (("799", 2.0), ("800", 2.5)):
    s = repl(s433, A433, '    result["factor"] = neutralize("signal", CHARS4, %.2f)\n' % frac, num)
    (D / ("%s_433上影线时点重心中性化%.1f.py" % (num, frac))).write_text(s, encoding="utf-8")
    print("写出", num)

# 801：788 换平滑时点重心
s788 = (D / "788_头部换手角加时点重心015.py").read_text(encoding="utf-8")
UTIME_TAIL = '    df["c_utime"] = -(_cen - 5.0)\n'
smooth = ('    _ord_u = df.sort_values(["instrument", "date"])\n'
          '    df["c_utime_s"] = _ord_u.groupby("instrument", sort=False)["c_utime"].transform(\n'
          '        lambda x: x.ewm(span=5, min_periods=3).mean()).reindex(df.index)\n')
s = repl(s788, UTIME_TAIL, UTIME_TAIL + smooth, "801_smooth")
m = re.search(r"    keep = \[(.*?)\]\n", s, re.S)
body = m.group(1).rstrip() + ', "c_utime_s"'
s = s[:m.start()] + "    keep = [%s]\n" % body + s[m.end():]
sig788 = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670", "c_utime"],\n'
          '                                  [1.00, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20, 0.15])\n')
sig801 = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670", "c_utime_s"],\n'
          '                                  [1.00, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20, 0.15])\n')
s = repl(s, sig788, sig801, "801_sig")
(D / "801_头部换手角加平滑时点重心015.py").write_text(s, encoding="utf-8"); print("写出 801")

# 802：766 + 时点重心 + 真换手
s = (D / "789_头部换手角加时点重心030.py").read_text(encoding="utf-8")
sig789 = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670", "c_utime"],\n'
          '                                  [1.00, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20, 0.30])\n')
sig802 = ('    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_indpeer_s", "c_conds670", "c_utime", "c_turnself"],\n'
          '                                  [1.00, 0.30, 0.15, 0.20, 0.15, 0.40, 0.20, 0.15, 0.15])\n')
s = repl(s, sig789, sig802, "802_sig")
(D / "802_头部换手角加时点重心真换手双源.py").write_text(s, encoding="utf-8"); print("写出 802")
