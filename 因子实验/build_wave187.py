# -*- coding: utf-8 -*-
"""第一百八十七批（对照项）：把 549 头里的挂单分量（c_ob 0.10）去掉，其余逐字节不变。

用途：1090（444 底座 + 低通 0.75）与 549（339 底座 + 挂单 0.10 + 低通 0.75）之间有两个差异
（中性化特征集、挂单分量），不能直接读成"底座升级"。1094 去掉挂单分量后与 1090 的头完全同构，
1090−1094 就是纯粹的中性化特征集差异；同时 1094−549 直接量出挂单分量在这一点的贡献。
"""
import io, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
O_COLS = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ob"]\n'
N_COLS = '    _cols = ["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body"]\n'
O_W = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10]), (len(result), 1))\n'
N_W = '    _W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15]), (len(result), 1))\n'
O_L = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.07])\n'
N_L = '    _W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10])\n'


def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)


t = load(549)
for a, b in ((O_COLS, N_COLS), (O_W, N_W), (O_L, N_L)):
    assert a in t, a
    t = t.replace(a, b, 1)
lines = t.split("\n")
assert lines[1].lstrip().startswith('"""'), lines[1]
lines.insert(2, "    【精选优化收尾对照】549 头去掉挂单分量（c_ob），其余逐字节不变；用于把手头的低通候选与 444 底座的差异拆成单变量。")
t = "\n".join(lines)
io.open(os.path.join(D, "1094_头部去挂单低通零点七五.py"), "w", encoding="utf-8").write(t)
print("wrote 1094_头部去挂单低通零点七五.py")
