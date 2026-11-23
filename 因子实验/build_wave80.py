# -*- coding: utf-8 -*-
"""第八十批：低通 alpha 细调 + 盘口深度/波动两个新源。

- 533：头部 + 低通 0.7（0.6 与 0.85 之间取点）。
- 534：头部 + 低通 0.6 + 挂单不平衡 0.10 + 中性化 2.0（524b 的权重减半版，试能否再降换手）。
- 535/536：把「盘口报价深度」「盘口不平衡波动」当分组变量（新信息源取证版，落在去均值族骨架上）。
"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
CODE = ROOT / "因子代码"

# 533：514 的 alpha 换 0.7
h514 = (CODE / "514_头部加日内信号低通平滑.py").read_text(encoding="utf-8")
assert "alpha=0.6" in h514
b533 = h514.replace("alpha=0.6", "alpha=0.7")
b533 = b533.replace('"""头部 + 日内信号 EWMA 低通平滑（换手杠杆，头部变体）。',
                    '"""头部 + 日内信号 EWMA 低通平滑（alpha=0.7，0.6~0.85 之间的细调点）。', 1)
(CODE / "533_头部加低通零点七.py").write_text(b533, encoding="utf-8")
print("写出 533_头部加低通零点七.py")

# 534：524 的挂单权重 0.15 → 0.10
h524 = (CODE / "524_头部加挂单不平衡低通二点零中性化.py").read_text(encoding="utf-8")
assert "_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15])" in h524
b534 = h524.replace("_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.15])",
                    "_W = np.tile(np.array([1.0, 0.30, 0.15, 0.20, 0.15, 0.10])")
b534 = b534.replace("_W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.10])",
                    "_W[_late] = np.array([1.0, 0.20, 0.10, 0.15, 0.10, 0.07])")
b534 = b534.replace('"""头部 + 一至三档挂单不平衡 0.15 + 中性化 2.0 倍 + 日内信号低通平滑（三个已验证杠杆全叠）。',
                    '"""头部 + 一至三档挂单不平衡 0.10 + 中性化 2.0 倍 + 日内低通 0.6（三杠杆的减权版）。', 1)
(CODE / "534_头部加挂单减权低通二点零中性化.py").write_text(b534, encoding="utf-8")
print("写出 534_头部加挂单减权低通二点零中性化.py")

# 535/536：沿用 485 骨架（含 1–3 档），把深度 / 盘口不平衡波动当分组变量
B = (CODE / "485_单独一至三档挂单不平衡反向.py").read_text(encoding="utf-8")
ANCHOR = '    df["c_max"] = -roll(win_ret, 40, "max", 10)'
KEEP_OLD = '    keep = ["r_30", "r_rv4", "r_rv8", "r_day", "c_size", "c_ivol", "c_illiq", "c_lowpx", "c_max", "c_upsh", "c_avgtr", "c_deals", "r_rv16", "c_body", "c_ob"]'
ASM = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["ladder"] = peer_ladder()
    result["signal"] = rank_blend(["ladder", "c_upsh", "c_avgtr", "r_rv16", "c_body", "c_ob"], [1.0, 0.30, 0.15, 0.20, 0.15, 0.15])
    result["factor"] = neutralize("signal", CHARS4, 1.5)'''
ASM_NEW = '''    CHARS4 = ["c_size", "c_ivol", "c_illiq", "c_lowpx"]
    result["signal"] = result["c_condrev"]
    result["factor"] = result["signal"]'''
for s in (ANCHOR, KEEP_OLD, ASM):
    assert s in B
RK = '    _rk66 = pd.concat([df.groupby("date")[c].rank(pct=True) for c in ["r_30", "r_rv4", "r_rv8", "r_day"]], axis=1).mean(axis=1) - 0.5'
COND = '    df["c_condrev"] = (_rk66 - _rk66.groupby([df["date"], _q66]).transform("mean"))  # 反转方向为正'
Q = '    _q66 = ((df.groupby("date")["%s"].rank(pct=True) * 3).clip(upper=2.999).fillna(-1)).astype(int)'
LAG = '''    _dd66 = df.groupby(["day", "instrument"], sort=False).agg(_x=("%s", "mean")).reset_index()
    _dd66 = _dd66.sort_values(["instrument", "day"]).reset_index(drop=True)
    _dd66["_x"] = _dd66.groupby("instrument", sort=False)["_x"].shift(1)
    _k66 = pd.MultiIndex.from_arrays([df["instrument"].values, df["day"].values])
    df["%s"] = pd.Series(_dd66.set_index(["instrument", "day"])["_x"].reindex(_k66).to_numpy(), index=df.index)'''

DEPTH = ('    df["_ob"] = _ob\n'
         '    # ===== 新增轴：盘口报价深度（1–3 档挂单总量，逐日滞后） =====\n'
         '    df["_depth"] = np.log1p(df["sum_bidv"] + df["sum_askv"])\n'
         + (LAG % ("_depth", "_depth_lag")) + "\n" + RK + "\n" + (Q % "_depth_lag") + "\n" + COND)

OBVOL = ('    df["_ob"] = _ob\n'
         '    # ===== 新增轴：盘口挂单不平衡的波动（40 窗口） =====\n'
         '    df["_obvol"] = roll(df["_ob"], 40, "std", 10)\n'
         + (LAG % ("_obvol", "_obvol_lag")) + "\n" + RK + "\n" + (Q % "_obvol_lag") + "\n" + COND)

for num, block, fname, desc in [
    ("535", DEPTH, "535_单独盘口报价深度分组条件反转.py", "盘口 1–3 档报价深度分组的条件反转"),
    ("536", OBVOL, "536_单独盘口不平衡波动分组条件反转.py", "盘口挂单不平衡波动分组的条件反转"),
]:
    body = B.replace(ANCHOR, ANCHOR + "\n" + block).replace(KEEP_OLD, KEEP_OLD[:-1] + ', "c_ob", "c_condrev"]').replace(ASM, ASM_NEW)
    body = body.replace('"""长窗口软同伴', '"""%s。\n\n    长窗口软同伴' % desc, 1)
    (CODE / fname).write_text(body, encoding="utf-8")
    print("写出", fname)
