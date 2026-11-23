# -*- coding: utf-8 -*-
"""第六十九批（491–494）：把"相关性极低但单独很弱"的慢轴按更高权重并入头部。

依据：452（已实现偏度单独）与 104 的截面秩相关只有 0.313、453（五档挂单不平衡单独）0.13 量级，
换手分别只有 0.2698 / 0.2248；但单独 IC_IR 0.664 / 0.238，都不到 0.8，所以不是"新信息源"，
而是**低换手叠加项**。本批测把它按 0.35 / 0.50 权重并入头部，看换手能不能换来 z ≤ −5 的可测改进。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"


def rep(t, old, new, tag):
    assert t.count(old) == 1, (tag, t.count(old))
    return t.replace(old, new, 1)


def set_w(s, tag, idx, wb, wl):
    """把分槽位权重向量里第 idx 个（0 基，最后一项）的分量改成 wb / wl。"""
    m = re.search(r'    _W = np\.tile\(np\.array\(\[([^\]]*)\]\), \(len\(result\), 1\)\)\n', s)
    assert m, tag + "_w"
    v = [x.strip() for x in m.group(1).split(",")]
    v[idx] = "%.2f" % wb
    s = s[:m.start(1)] + ", ".join(v) + s[m.end(1):]
    m = re.search(r'    _W\[_late\] = np\.array\(\[([^\]]*)\]\)\n', s)
    assert m, tag + "_wl"
    v = [x.strip() for x in m.group(1).split(",")]
    v[idx] = "%.2f" % wl
    return s[:m.start(1)] + ", ".join(v) + s[m.end(1):]


B456 = D / "456_头部加偏度二零点二.py"
B457 = D / "457_头部加偏度与挂单不平衡.py"


def c491(s):   # 头部 + 偏度 0.35
    return set_w(s, "491", -1, 0.35, 0.15)


def c492(s):   # 头部 + 偏度 0.50（探针）
    return set_w(s, "492", -1, 0.50, 0.20)


def c493(s):   # 头部 + 偏度 0.35（挂单保持 0.15）
    return set_w(s, "493", -2, 0.35, 0.15)


def c494(s):   # 头部 + 偏度 0.35 + 挂单不平衡提高到 0.25
    s = set_w(s, "494a", -2, 0.35, 0.15)
    return set_w(s, "494b", -1, 0.25, 0.15)


CANDS = [("491", "头部加偏度零点三五", B456, c491),
         ("492", "头部加偏度零点五零探针", B456, c492),
         ("493", "头部加偏度零点三五挂单零点一五", B457, c493),
         ("494", "头部加偏度零点三五挂单零点二五", B457, c494)]

if __name__ == "__main__":
    for num, name, base, fn in CANDS:
        s = fn(base.read_text(encoding="utf-8"))
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p.name)
