# -*- coding: utf-8 -*-
"""第十四批（197+）：把本轮已单独验证过的杠杆尽量合到一起，并试一个双同伴来源的组合。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import add_rv16, keep, tail, doc  # noqa: E402
from build_wave10 import base  # noqa: E402
from build_wave5 import add_exposure, add_feats  # noqa: E402
from build_wave13 import add_bigvw_sql, SNIP_ADD_SCALE  # noqa: E402

LADDER_OLD = '        for c in ["r_30", "r_rv4", "r_rv8", "r_rv16", "r_day"]:'
LADDER_NEW = '        for c in ["r_30", "r_rv4", "r_rv8", "r_rv16", "r_bv4", "r_bv8", "r_day"]:'

ANCHOR = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'

SPECS = [
    ("因子代码/197_五杠杆叠加.py", """五杠杆叠加：长窗口同伴（240 日）+ 六尺度 + 大单成本锚尺度 + 14:30 首半小时动量
+ 四特征 1.5 倍中性化（组合，09:30 修复版）。

    把本轮**单独验证过有增量**的构造放在一个因子里：144（同伴窗口 240 日）、119/142（约 2 日尺度，换手最优）、
      195（大单成本锚尺度）、147（仅 14:30 截面生效的首半小时动量）。
      各附加项权重从 0.15 降到 0.10，避免单一方向主导。这是「多条独立信息源合成」的极端版本，
      用于检验叠加是否有上限（对照 181 的四杠杆版本）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
]

if __name__ == "__main__":
    for path, text in SPECS:
        s = base()
        s = add_rv16(s)
        s = add_bigvw_sql(s)
        s = add_exposure(s)
        s = add_feats(s, ("c_m30",))
        assert s.count(ANCHOR) == 1
        s = s.replace(ANCHOR, ANCHOR + SNIP_ADD_SCALE)
        assert s.count(LADDER_OLD) == 1
        s = s.replace(LADDER_OLD, LADDER_NEW)
        s = keep(s, ("r_bv4", "r_bv8", "c_m30"))
        s = tail(s, ("c_m30",), (0.10,))
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
