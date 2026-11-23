# -*- coding: utf-8 -*-
"""第一百八十六批（精选优化收尾探针）：把「日内低通」这条公榜已证明有效的轴，
架到三条候选线里唯一有三窗口证据的 444 底座（前日行业中性化）上。

此前低通只在 339 系底座（549/1062/1065/1076）与「低通×2.5×」（1063/1064）上测过，
444 底座（1082=2.0×、1083=2.5×）从未加过低通。本批是**单参数改动**（只加一段 ewm 平滑），
用来回答：IC 均值维度领先的 444 底座能否再吃到低通杠杆，还是两者仍是同一条直线。

- 1090：1082（444 底座 2.0×）+ 低通 alpha 0.750
- 1091：1082（444 底座 2.0×）+ 低通 alpha 0.850
- 1092：1083（444 底座 2.5×）+ 低通 alpha 0.750
"""
import io, os

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")
ANCHOR = '    result["signal"] = pd.Series(np.where(_den > 0, _num / np.where(_den > 0, _den, 1.0), np.nan), index=result.index)\n'
LOWPASS = (
    '    _ord = result.sort_values(["instrument", "date"])\n'
    '    _ew = _ord.groupby("instrument", sort=False)["signal"].transform(lambda x: x.ewm(alpha=%s, adjust=False).mean())\n'
    '    result["signal"] = _ew.reindex(result.index)\n'
)


def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)


def save(num, name, text):
    io.open(os.path.join(D, "%d_%s.py" % (num, name)), "w", encoding="utf-8").write(text)
    print("wrote %d_%s.py" % (num, name))


def with_lowpass(t, alpha, note):
    assert ANCHOR in t, "no signal anchor"
    t = t.replace(ANCHOR, ANCHOR + LOWPASS % alpha, 1)
    lines = t.split("\n")
    assert lines[1].lstrip().startswith('"""'), lines[1]
    lines.insert(2, "    %s" % note)
    return "\n".join(lines)


N = "【精选优化收尾探针】444 前日行业中性化底座（%s）× 日内低通 alpha %s；只加一段 ewm 平滑，其它逐字节不变。"

save(1090, "行业中性化前日行业二点零加低通零点七五", with_lowpass(load(1082), "0.750", N % ("2.0 倍", "0.750")))
save(1091, "行业中性化前日行业二点零加低通零点八五", with_lowpass(load(1082), "0.850", N % ("2.0 倍", "0.850")))
save(1092, "行业中性化前日行业二点五加低通零点七五", with_lowpass(load(1083), "0.750", N % ("2.5 倍", "0.750")))
