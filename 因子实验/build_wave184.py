# -*- coding: utf-8 -*-
"""第一百八十四批：c_hitime 权重上扫与跨臂加码（1035/1036/1034 已显示权重 0.30 远优于 0.15）。
- 1042/1043/1044：833 臂 hitime 0.45 / 0.60 / 0.75
- 1045/1046：925 臂 hitime 0.30 / 0.45
- 1047：960 臂 hitime 0.30
- 1048：866 压力臂 hitime 0.30
- 1049：833 臂 hitime 0.30 + 1.75 倍中性化
"""
import io, os, re

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")

def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)

def save(num, name, text):
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write(text)
    print("wrote %d_%s.py" % (num, name))

def sub(t, old, new):
    assert old in t, old[:120]
    return t.replace(old, new, 1)

def set_last_weight(t, w):
    m = re.search(r'(    result\["signal"\] = rank_blend\(\[.*?\], \[)([^\]]*?)(\]\))', t)
    assert m, "no rank_blend"
    ws = m.group(2).split(", ")
    ws[-1] = str(w)
    return t[:m.start(2)] + ", ".join(ws) + t[m.end(2):]

save(1042, "最高价时段833权重045", set_last_weight(load(1035), "0.45"))
save(1043, "最高价时段833权重060", set_last_weight(load(1035), "0.60"))
save(1044, "最高价时段833权重075", set_last_weight(load(1035), "0.75"))
save(1045, "最高价时段925权重030", set_last_weight(load(1034), "0.30"))
save(1046, "最高价时段925权重045", set_last_weight(load(1034), "0.45"))
save(1047, "最高价时段960权重030", set_last_weight(load(1036), "0.30"))
save(1048, "最高价时段866权重030", set_last_weight(load(1003), "0.30"))
save(1049, "最高价时段833权重030一点七五", sub(load(1035), 'neutralize("signal", CHARS4, 2.0)', 'neutralize("signal", CHARS4, 1.75)'))
