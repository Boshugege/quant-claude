# -*- coding: utf-8 -*-
"""第二十七批（244+）：上影线信号的定义细化（在 233 = 长窗口同伴阶梯 + 上影线 0.30 的骨架上）。

第二十六批（239–243）结论：把 154/155/156 的影线不对称/实体占比/摆动比率杠杆都搬到长窗口骨架后，
**都打不过单纯的上影线 0.30**（239 +0.024/z=1.25、240 单独影线不对称 +0.049/z=2.79、241 实体占比 +0.065/z=3.38），
说明这些微结构杠杆彼此高度替代。本批改为**细化上影线本身的定义**：
近端加权（EWMA）、更短窗口、替换成"窗口内冲高回落"（收-高）、以及与实体占比的更重组合。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "因子代码/233_长上影线附加三成.py"
UPSH = '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh"]\n'
SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n'

FADE = ('    # 窗口内冲高回落：Σ(窗口最高-窗口末价) / Σ(窗口最高-窗口最低)，40 窗口均值取负。\n'
        '    df["c_fade"] = -roll((df["win_high"] - df["last_px"]) / (df["win_high"] - df["win_low"]).where(df["win_high"] > df["win_low"]), 40, "mean", 10)\n')
EWMA = ('    # 近端加权上影线：日级上影线占比的 20 期 EWMA（近端权重更大），取负。\n'
        '    _upsh = df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0)\n'
        '    df["c_upsh"] = -_upsh.groupby(df["instrument"], sort=False).transform(lambda x: x.ewm(span=20, min_periods=10).mean())\n')
BODY = ('    # 实体占比（加重）：窗口内 Σ|收-开| / Σ(高-低)，40 窗口均值取负。\n'
        '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')

# (编号, 名称, [(旧片段, 新片段), ...])
SPECS = [
    ("244", "长上影线加十四点半动量", [
        (SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_m30"], [1.0, 0.30, 0.15])\n')]),
    ("245", "长冲高回落三成", [
        (UPSH, FADE),
        (KEEP, KEEP.replace('"c_upsh"]', '"c_fade"]')),
        (SIG, '    result["signal"] = rank_blend(["ladder", "c_fade"], [1.0, 0.30])\n')]),
    ("246", "长上影线EWMA加权", [
        (UPSH, EWMA)]),
    ("247", "长上影线加实体占比三成", [
        (UPSH, UPSH + BODY),
        (KEEP, KEEP.replace('"c_upsh"]', '"c_upsh", "c_body"]')),
        (SIG, '    result["signal"] = rank_blend(["ladder", "c_upsh", "c_body"], [1.0, 0.30, 0.30])\n')]),
    ("248", "长上影线窗口二十", [
        (UPSH, UPSH.replace("), 40, \"mean\", 10)", "), 20, \"mean\", 10)"))]),
]

def main():
    s0 = BASE.read_text(encoding="utf-8")
    for num, name, reps in SPECS:
        s = s0
        for old, new in reps:
            assert s.count(old) == 1, (num, old[:60])
            s = s.replace(old, new, 1)
        p = ROOT / ("因子代码/%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)

if __name__ == "__main__":
    main()
