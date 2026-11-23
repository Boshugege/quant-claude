# -*- coding: utf-8 -*-
"""第一百五十五批：逐维度优化（IC_IR / 压力 / 多空 / 换手），并补两个风格条件反转新源。
- 863：833 时点重心 0.45 -> 0.50（IC_IR 上探）
- 864：833 时点重心 0.45 -> 0.55（IC_IR 上探上限）
- 865：842（时点重心0.45+跳空0.15）跳空 -> 0.20（换手/压力平衡）
- 866：842 时点重心 -> 0.30（等价 816 管线加跳空0.15，压力维度）
- 867：842 时点重心 -> 0.20（等价 823 多空臂加轻度时点重心，多空维度）
- 868：828（市值加权行业滞后+时点重心0.30）时点重心 -> 0.45（多空/IC_IR）
- 869：833 + 官方 SIZE 前一日五分位内条件反转（新源）
- 870：833 + 官方 LIQUIDTY 前一日五分位内条件反转（新源）
"""
import io, os, re

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "因子代码")

def load(num):
    for f in sorted(os.listdir(D)):
        if f.startswith(str(num) + "_") and f.endswith(".py"):
            return f, io.open(os.path.join(D, f), encoding="utf-8").read()
    raise SystemExit("missing base %d" % num)

def save(num, name, text):
    path = os.path.join(D, "%d_%s.py" % (num, name))
    io.open(path, "w", encoding="utf-8").write(text)
    print("wrote", os.path.basename(path))

def sub_once(text, old, new):
    assert old in text, old
    return text.replace(old, new, 1)

# ---------- 863 / 864：IC_IR 网格 ----------
_, b833 = load(833)
OLD_UT = 'result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45])'
for num, name, w in [(863, "行业市值中性化加时点重心050", "0.50"),
                     (864, "行业市值中性化加时点重心055", "0.55")]:
    t = sub_once(b833, OLD_UT, OLD_UT.replace("0.15, 0.45])", "0.15, %s])" % w))
    save(num, name, t)

# ---------- 865：换手/压力 ----------
_, b842 = load(842)
OLD_G = 'result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "c_gap"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.15])'
save(865, "行业市值中性化时点重心045跳空020", sub_once(b842, OLD_G, OLD_G.replace("0.45, 0.15])", "0.45, 0.20])")))

# ---------- 866 / 867：压力 / 多空（时点重心权重下调） ----------
save(866, "行业市值中性化时点重心030跳空015", sub_once(b842, OLD_G, OLD_G.replace("0.15, 0.45, 0.15])", "0.15, 0.30, 0.15])")))
save(867, "行业市值中性化时点重心020跳空015", sub_once(b842, OLD_G, OLD_G.replace("0.15, 0.45, 0.15])", "0.15, 0.20, 0.15])")))

# ---------- 868：多空臂加时点重心 0.45 ----------
_, b828 = load(828)
t = b828
t = sub_once(t, "_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10, 0.30]), (len(result), 1))",
                "_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10, 0.45]), (len(result), 1))")
t = sub_once(t, "_W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08, 0.20])",
                "_W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.08, 0.30])")
save(868, "市值加权行业滞后加时点重心045", t)

# ---------- 869 / 870：风格条件反转（新源） ----------
COND_BLOCK = '''    # ===== 新轴：按前一日官方风格暴露分组的条件反转（五分位内去均值） =====
    _rkS = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5
    _qS = pd.qcut(pd.Series(df["{col}"], index=df.index).groupby(df["date"]).rank(pct=True), 5, labels=False, duplicates="drop")
    df["{name}"] = _rkS - _rkS.groupby([df["date"], _qS]).transform("mean")

'''
ANCHOR = "    df = df.replace([np.inf, -np.inf], np.nan)\n    keep = "
for num, name, col, tag in [(869, "行业市值中性化时点重心加风格SIZE条件反转", "s_SIZE", "c_condsize"),
                            (870, "行业市值中性化时点重心加风格LIQUIDTY条件反转", "s_LIQUIDTY", "c_condliq")]:
    t = b833
    t = sub_once(t, ANCHOR, COND_BLOCK.format(col=col, name=tag) + ANCHOR)
    t = sub_once(t, '"c_body", "c_utime"]\n', '"c_body", "c_utime", "%s"]\n' % tag)
    t = sub_once(t, '["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45]',
                    '["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_utime", "%s"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.45, 0.30]' % tag)
    save(num, name, t)
