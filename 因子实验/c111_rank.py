# -*- coding: utf-8 -*-
"""用 c110 的两步模型给评分机的本地结果排序（c96_collect 的输出）。用法：python3 c111_rank.py <c96 输出文件>"""
import contextlib
import io
import re
import sys
import numpy as np
sys.argv = [sys.argv[0], "因子实验/public_rows.json", sys.argv[1]]
src = open("因子实验/c110_public_model.py", encoding="utf-8").read()
g = {}
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(src, "c110", "exec"), g)
rows = []
for line in open(sys.argv[2], encoding="utf-8"):
    m = re.match(r"(\S+)\s+[\d.]+\s+([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)", line.strip())
    if not m:
        continue
    ic, ir, ls, st, to = map(float, m.groups()[1:])
    loc = (ic, ir, to, ls, st)
    pub = g["to_public"](loc)
    iso = float(g["predict"](g["m"], pub[None, :])[0]); smo = float(g["logi"](g["TH"], pub[None, :])[0])
    rows.append(((iso + smo) / 2, iso, smo, m.group(1), loc, pub))
rows.sort(key=lambda r: -r[0])
print("%-34s %6s %6s %6s | 本地 IC／IC_IR／换手 | 估公榜 IC／IC_IR／换手／压力" % ("任务", "平均", "保序", "平滑"))
for avg, iso, smo, nm, loc, pub in rows[:30]:
    print("%-34s %.4f %.4f %.4f | %.4f／%.3f／%.3f | %.4f／%.3f／%.3f／%.3f" % (nm[:34], avg, iso, smo, loc[0], loc[1], loc[2], pub[0], pub[1], pub[2], pub[4]))
