# -*- coding: utf-8 -*-
"""第一百零七批：条件反转族的时间平滑版（639 BETA / 640 SIZE），复刻同伴族的低换手做法。"""
import pathlib
ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"
def wr(n, s): (CODE / n).write_text(s, encoding="utf-8"); print("写出", n)
OLD = '    result["signal"] = result["c_condrev"]'
NEW = ('    _ordc = result.sort_values(["instrument", "date"])\n'
       '    result["signal"] = _ordc.groupby("instrument", sort=False)["c_condrev"].transform(\n'
       '        lambda x: x.ewm(span=10, min_periods=5).mean()).reindex(result.index)')
for src, dst, desc in [
    ("623_单独按BETA五分位分组的条件反转.py", "639_单独按BETA五分位分组的条件反转平滑版.py", "BETA"),
    ("624_单独按SIZE五分位分组的条件反转.py", "640_单独按SIZE五分位分组的条件反转平滑版.py", "SIZE"),
]:
    t = (CODE / src).read_text(encoding="utf-8")
    assert OLD in t, src
    t = t.replace(OLD, NEW, 1)
    t = t.replace(f"（第 t 个截面上把多尺度反转排名在 {desc} 五分位内去均值；暴露表错一天）",
                  f"（第 t 个截面上把多尺度反转排名在 {desc} 五分位内去均值；再做 10 窗口 EWMA 平滑；暴露表错一天）", 1)
    wr(dst, t)
