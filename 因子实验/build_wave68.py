# -*- coding: utf-8 -*-
"""第六十八批（485–488）：盘口族改成显式 1–3 档，消除"本地 4–5 档为 NULL"的测量偏差。

背景：评测机告警「SQL 用到了 4–5 档盘口；本地数据只有 3 档，4–5 档为 NULL，结果与平台不同」。
本地把 4–5 档 COALESCE 成 0，等价于只用 1–3 档；平台上有全部 5 档，所以已跑的 421/422/435/437/438/453/457
在评测机上测到的其实是"1–3 档版"，与平台交付版不是同一个因子。
本批把 SQL 里 4–5 档整列删掉，交付版与本地测量口径完全一致（本地分数不变，但结果可解释、无告警）。
"""
import pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
D = ROOT / "因子代码"

DROP = [r"            CAST\(COALESCE\(bid_volume[45], 0\) AS DOUBLE\) AS bv[45],\n",
        r"            CAST\(COALESCE\(ask_volume[45], 0\) AS DOUBLE\) AS av[45],\n"]


def to_l13(s, tag):
    n = 0
    for pat in DROP:
        while re.search(pat, s):
            s = re.sub(pat, "", s, count=1)
            n += 1
    assert n == 4, (tag, n)
    a = s.count("bv1 + bv2 + bv3 + bv4 + bv5")
    b = s.count("av1 + av2 + av3 + av4 + av5")
    assert a >= 1 and b >= 1, (tag, a, b)
    s = s.replace("bv1 + bv2 + bv3 + bv4 + bv5", "bv1 + bv2 + bv3")
    s = s.replace("av1 + av2 + av3 + av4 + av5", "av1 + av2 + av3")
    assert "bid_volume4" not in s and "ask_volume5" not in s, tag
    return s


CANDS = [("422", "485", "单独一至三档挂单不平衡反向"),
         ("435", "486", "头部加一至三档挂单不平衡"),
         ("437", "487", "单独一至三档报价深度"),
         ("438", "488", "单独一至三档盘口波动")]

if __name__ == "__main__":
    for src, num, name in CANDS:
        fs = sorted(D.glob("%s_*.py" % src))
        assert len(fs) == 1, (src, fs)
        s = to_l13(fs[0].read_text(encoding="utf-8"), num)
        p = D / ("%s_%s.py" % (num, name))
        p.write_text(s, encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "因子实验/make_ipynb.py"), str(p)], check=True)
        print("写出", p.name)
