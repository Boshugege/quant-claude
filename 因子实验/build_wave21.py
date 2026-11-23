# -*- coding: utf-8 -*-
"""第二十六批（239+）：把 154/155/156 的分钟微结构杠杆（影线不对称、实体占比、摆动比率）
搬到已验证的长窗口同伴骨架上（233 = 176 骨架 + 上影线 0.30，平台 +0.0695/z=3.46）。

154/155/156 之前只在短窗口（103/104 骨架）上试过；单独成因子时 154（影线不对称）在旧口径下
screen 得到 +0.0546/z=3.03。本批把它们与已验证的上影线杠杆组合，看是否叠加增益。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE = ROOT / "因子代码/233_长上影线附加三成.py"
UPSH = '    df["c_upsh"] = -roll(df["sum_upsh"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n'
KEEP = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh"]\n'
SIG = '    result["signal"] = rank_blend(["ladder", "c_upsh"], [1.0, 0.30])\n'

SHADOW = ('    # 分钟影线不对称：窗口内 Σ(上影-下影) / Σ(全幅)，最近 40 窗口均值取负'
          '（下影线越长，买盘承接越强，反转越强）。\n'
          '    df["c_shadow"] = -roll((df["sum_upsh"] - df["sum_dnsh"]) / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')
BODY = ('    # 分钟实体占比：窗口内 Σ|收-开| / Σ(高-低)，最近 40 窗口均值取负'
        '（实体占比越大，单边趋势越强，反转越弱）。\n'
        '    df["c_body"] = -roll(df["sum_body"] / df["sum_rng"].where(df["sum_rng"] > 0), 40, "mean", 10)\n')
PATH = ('    # 摆动比率：窗口内 Σ全幅 / (窗口最高-最低)，最近 40 窗口均值取负'
        '（路径越长越反复，反转越强）。\n'
        '    _path = df["sum_rng"] / (df["win_high"] - df["win_low"]).where(df["win_high"] > df["win_low"])\n'
        '    df["c_path"] = -roll(_path, 40, "mean", 10)\n')

# (编号, 名称, 追加分量代码, keep 追加, 信号列, 权重表)
SPECS = [
    ("239", "长上影线加影线不对称", SHADOW, ["c_shadow"], ["ladder", "c_upsh", "c_shadow"], [1.0, 0.30, 0.15]),
    ("240", "长影线不对称三成",     SHADOW, ["c_shadow"], ["ladder", "c_shadow"], [1.0, 0.30]),
    ("241", "长上影线加实体占比",   BODY,   ["c_body"],   ["ladder", "c_upsh", "c_body"],   [1.0, 0.30, 0.15]),
    ("242", "长上影线加摆动比率",   PATH,   ["c_path"],   ["ladder", "c_upsh", "c_path"],   [1.0, 0.30, 0.15]),
    ("243", "长上影线加影线不对称二成五", SHADOW, ["c_shadow"], ["ladder", "c_upsh", "c_shadow"], [1.0, 0.30, 0.25]),
]

def main():
    s0 = BASE.read_text(encoding="utf-8")
    assert s0.count(UPSH) == 1 and s0.count(KEEP) == 1 and s0.count(SIG) == 1
    for num, name, comp, keepadd, cols, wts in SPECS:
        s = s0
        assert s.count(comp) == 0
        s = s.replace(UPSH, UPSH + comp, 1)
        newkeep = KEEP.replace('"c_upsh"]', '"c_upsh", %s]' % ", ".join('"%s"' % k for k in keepadd))
        s = s.replace(KEEP, newkeep, 1)
        newsig = '    result["signal"] = rank_blend([%s], [%s])\n' % (
            ", ".join('"%s"' % c for c in cols), ", ".join("%s" % w for w in wts))
        s = s.replace(SIG, newsig, 1)
        p = ROOT / ("因子代码/%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p)

if __name__ == "__main__":
    main()
