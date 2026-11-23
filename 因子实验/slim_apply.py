# -*- coding: utf-8 -*-
"""把 561 派生文件里「纯粹为 ladder 服务的月度同伴相关矩阵」整块删掉，并缩短回看窗口。

这些新源探针的信号（result["signal"] = result["c_xxx"]）根本不用 ladder，
而月度相关的 corr() 是整个任务里最贵的一段，删掉后单任务耗时可降数倍。
只对「信号不使用 ladder / peer_mean」的文件用。
"""
import io, pathlib, sys

START = "    # 同伴：每个自然月月初，用此前 PEER_LOOKBACK 个交易日的日收益（收盘段价格 / 昨收）计算相关系数。\n"
END = "        peers_by_month[m] = (idx, full_w)\n"
LADDER = '    result["ladder"] = peer_ladder()\n'
LOOK_OLD = "    LOOKBACK_DAYS = 400\n"
LOOK_NEW = "    LOOKBACK_DAYS = 90\n"

def slim(text):
    i = text.index(START)
    j = text.index(END, i) + len(END)
    text = text[:i] + text[j:]
    assert LADDER in text
    text = text.replace(LADDER, "", 1)
    assert LOOK_OLD in text
    text = text.replace(LOOK_OLD, LOOK_NEW, 1)
    return text

for fn in sys.argv[1:]:
    p = pathlib.Path(fn)
    s = p.read_text(encoding="utf-8")
    if "# 同伴：每个自然月月初" not in s:
        print("跳过（已是瘦身版或非 561 派生）:", fn); continue
    p.write_text(slim(s), encoding="utf-8")
    print("瘦身", fn)
