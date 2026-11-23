# -*- coding: utf-8 -*-
"""由 build_1630 生成"1617 + 规模比 + 缓冲带"，再把缓冲带改成"最小抬升"（10-22）：
腿仍按缓冲后的排名定（上一截面同日在多头／空头腿的股票加／减 β），但输出用原值；腿内低于腿外最高者的股票只抬到它之上 m 个标准差
（空头腿对称），其余股票原值不动。本地（评分机 c105g，2023–24）同一换手下五项都比原缓冲带好。
用法：python3 因子实验/build_1649.py <编号> <名字> <buf> <m> [<q_w>]"""
import subprocess
import sys

num, title, buf, lm = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
qw = sys.argv[5] if len(sys.argv) > 5 else "0.3"
subprocess.run([sys.executable, "因子实验/build_1630.py", num, title, "0", buf, "0", qw, "s1", "0.3"], check=True)
DST = "因子代码/%s_%s.py" % (num, title)
s = open(DST, encoding="utf-8").read()


def rep(a, b):
    global s
    assert s.count(a) == 1, (s.count(a), a[:80])
    s = s.replace(a, b)


rep("_POST_EW, _POST_BUF, _POST_ANCHOR_W, _POST_Q_W, _POST_Q_MODE, _POST_Q_SM = ",
    "_POST_LIFT = %r                                                                    # 最小抬升的间隔 m（标准差）\n_POST_EW, _POST_BUF, _POST_ANCHOR_W, _POST_Q_W, _POST_Q_MODE, _POST_Q_SM = " % lm)
rep("""            vv = res[idx]
            if prev_day is not None and dd == prev_day and prev_leg:
                vv = vv + _POST_BUF * sub["instrument"].map(prev_leg).fillna(0.0).to_numpy(float)
            rr = pd.Series(vv).rank(pct=True).to_numpy()
            res[idx] = vv
""", """            z0 = res[idx].copy()
            vv = z0
            if prev_day is not None and dd == prev_day and prev_leg:
                vv = vv + _POST_BUF * sub["instrument"].map(prev_leg).fillna(0.0).to_numpy(float)
            rr = pd.Series(vv).rank(pct=True).to_numpy()
            L, S = rr > 0.9, rr <= 0.1                                           # 最小抬升：腿按缓冲后的排名定，输出用原值 z0，
            o = z0.copy()                                                        # 腿内低于腿外最高者的股票只抬到它之上 _POST_LIFT 个标准差
            if L.any() and (~L).any():
                cL = z0[~L].max() + _POST_LIFT
                o = np.where(L, np.maximum(z0, cL + 1e-6 * (z0 - z0.min())), o)
            if S.any() and (~S).any():
                cS = z0[~S].min() - _POST_LIFT
                o = np.where(S, np.minimum(z0, cS - 1e-6 * (z0.max() - z0)), o)
            res[idx] = o
""")
rep("腿边界缓冲带 β=", "腿边界缓冲带（最小抬升 m=%.1f：腿按缓冲后的排名定，输出用原值，腿内股票只抬到腿外最高者之上 m 个标准差）β=" % lm)
open(DST, "w", encoding="utf-8").write(s)
print("改为最小抬升", DST)
