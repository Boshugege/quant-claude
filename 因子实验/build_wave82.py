# -*- coding: utf-8 -*-
"""第八十二批：低通档内细调 + 两个尚未用过的成交结构分组（平均单笔成交额 / 成交笔数）。"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

# 543/544：534（低通 0.6 + 挂单 0.10 + 2.0，公榜估计 0.8438 最高）的 alpha 细调
h534 = (CODE / "534_头部加挂单减权低通二点零中性化.py").read_text(encoding="utf-8")
assert "alpha=0.6" in h534
for num, alpha, fname in [("543", "0.55", "543_头部加挂单减权低通零点五五.py"),
                          ("544", "0.70", "544_头部加挂单减权低通零点七.py")]:
    body = h534.replace("alpha=0.6", "alpha=%s" % alpha)
    body = body.replace('"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.6（三杠杆的减权版）。',
                        '"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 %s（低通档内细调）。' % alpha, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)

# 541/542：沿用 497 的去均值族骨架，换两个没试过的成交结构分组
S = (CODE / "497_单独行业内条件反转.py").read_text(encoding="utf-8")
OLD = '''    _q66 = df["ind_code"]
    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 468 实测方向为正'''
assert OLD in S
RK = '    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5'
Q = '    _q66 = ((df.groupby("date")["%s"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'
COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'
for num, col, fname, desc in [
    ("541", "c_avgtr", "541_单独平均单笔成交额分组条件反转.py", "平均单笔成交额（大小单结构代理）分组的条件反转"),
    ("542", "c_deals", "542_单独成交笔数分组条件反转.py", "成交笔数（活跃度代理）分组的条件反转"),
]:
    ANCHOR = '    df["c_max"] = -roll(win_ret, 40, "max", 10)'
    assert ANCHOR in S
    # c_avgtr/c_deals 在 OLD 那一行之后才计算，所以真正的分组块必须插到 c_max 之后
    body = S.replace(OLD, '    df["c_condrev"] = np.nan  # 占位：真正的分组在 c_avgtr/c_deals 计算之后')
    body = body.replace(ANCHOR, ANCHOR + "\n" + RK + "\n" + (Q % col) + "\n" + COND)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
