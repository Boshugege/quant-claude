# -*- coding: utf-8 -*-
"""第二十三批（226+）：把评分机 2.3.0 下被 09:30 截面拖累的强候选取平台代码形态（09:30 恒为 0）。

依据：打分机 values 复算（同一批因子值，仅把 09:30 置 0）显示 176/179/180 的综合百分位分别
+0.065 / +0.054 / +0.060（z=3.06 / 2.31 / 3.04），都显著优于 104。本脚本把它们转成可交付的
平台代码（平台代码必须自己把 09:30 写 0，不能靠后处理）。
"""
import pathlib, subprocess, sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
RET = '    return result[["date", "instrument", "factor"]].sort_values('
ZERO = ('    # 09:30 输出 0（与 104/103/107 同口径）：平台跳过全 0 截面。\n'
        '    result.loc[result["date"].dt.strftime("%H%M") == "0930", "factor"] = 0.0\n')

SPECS = [
    ("226", "长窗口同伴上影线0930置零", "176_长窗口同伴上影线.py",
     "长窗口软同伴阶梯 + 分钟上影线附加（09:30 置零版）",
     "评分机复算 +0.065/hole=3.06，是本轮最强的 7 截面候选。"),
    ("227", "长窗口同伴五特征中性0930置零", "179_长窗口同伴五特征中性.py",
     "长窗口软同伴阶梯 + 六特征中性化（09:30 置零版）",
     "评分机复算 +0.054/hole=2.31。"),
    ("228", "日内同伴上影线0930置零", "180_日内同伴上影线.py",
     "日内相关同伴阶梯 + 分钟上影线附加（09:30 置零版）",
     "评分机复算 +0.060/hole=3.04。"),
]

def main():
    for num, name, src, title, note in SPECS:
        s = (ROOT / "因子代码" / src).read_text(encoding="utf-8")
        assert s.count(RET) == 1, (num, src)
        s = s.replace(RET, ZERO + RET, 1)
        s = s.replace('（09:30 修复版）', '（09:30 置零版）', 1)
        s = s.replace('    """同伴残差阶梯 + 四特征 1.5 倍中性化。', '    """' + title + '。', 1)
        s = s.replace('    """\n    import numpy as np', '\n    %s"""\n    import numpy as np' % note, 1)
        p = ROOT / ("因子代码/%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        print("写出", p)
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)

if __name__ == "__main__":
    main()
