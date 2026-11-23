# -*- coding: utf-8 -*-
"""生成 121（窗口买卖压力附加）与 122（上涨分钟占比附加）两个平台因子代码。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, soft_peer, keep, tail, doc  # noqa: E402

PA_SNIPPET = '''    # 窗口收盘价相对本窗口成交 VWAP 的位置（买/卖压力不对称），最近 40 个窗口的均值。
    _pa = safe_log_ratio(px, vwap_of(df["sum_amt"], df["sum_vol"]))
    df["c_pa"] = roll(_pa, 40, "mean", 10)
'''

UP_SNIPPET = '''    # 窗口内上涨分钟占比（1 分钟尺度买卖力量），取负：买压越强，后续反转越明显。
    _up = df["n_up"] / df["n_min"].where(df["n_min"] > 0)
    df["c_up"] = -roll(_up, 40, "mean", 10)
'''


def anchor(src):
    a = '    df["c_lowpx"] = -np.log(px.where(px > 0))\n'
    assert src.count(a) == 1
    return a


def add_path_asym(src):
    a = anchor(src)
    return src.replace(a, a + PA_SNIPPET)


def add_up_ratio(src):
    a = "            CAST(COALESCE(close, 0) AS DOUBLE) AS px_close,"
    assert src.count(a) == 1
    src = src.replace(a, a + "\n            CAST(COALESCE(open, 0) AS DOUBLE) AS px_open,")
    b = "WHEN hms > '090000' AND hms <= '093000' THEN 93000"
    assert src.count(b) == 1
    src = src.replace(b, "WHEN px_close > 0 AND px_open > 0 AND px_close > px_open THEN 1 ELSE 0 END AS up_min,\n            CASE\n                " + b)
    c = "            SUM(vol_c) AS sum_vol,"
    assert src.count(c) == 1
    src = src.replace(c, c + "\n            SUM(up_min) AS n_up,\n            COUNT(up_min) AS n_min,")
    e = "        f.sum_amt, f.sum_vol, f.last_px, f.pre_close"
    assert src.count(e) == 1
    src = src.replace(e, "        f.sum_amt, f.sum_vol, f.last_px, f.n_up, f.n_min, f.pre_close")
    f = '    for c in ["sum_amt", "sum_vol", "last_px", "pre_close"]:'
    assert src.count(f) == 1
    src = src.replace(f, '    for c in ["sum_amt", "sum_vol", "last_px", "n_up", "n_min", "pre_close"]:')
    a = anchor(src)
    return src.replace(a, a + UP_SNIPPET)


SPECS = {
    "因子代码/121_软同伴窗口VWAP压力附加.py": (
        add_path_asym, ("c_pa",), (0.15,),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 窗口买卖压力附加（新方向，09:30 修复版）。

    新增“窗口收盘价相对本窗口成交 VWAP 的位置” ln(窗口末价/窗口 VWAP) 最近 40 个窗口的均值，权重 0.15。
    含义：窗口内系统性收在 VWAP 之上说明承接买盘强、属过度反应，后续回补更明显。
    本地粗筛（2023 验证集，localeval 口径）：IC 0.0774→0.0790、IC_IR 1.619→1.638，换手基本不变。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
    "因子代码/122_软同伴上涨分钟占比附加.py": (
        add_up_ratio, ("c_up",), (0.15,),
        """软同伴残差阶梯 + 四特征 1.5 倍中性化 + 上涨分钟占比附加（新方向，09:30 修复版）。

    新增 1 分钟尺度信号：最近 40 个窗口内“收盘价高于开盘价的分钟数占比”的均值，取负（买压越强后续越弱）。
    这是本项目第一次使用分钟级的窗口内结构（此前所有因子只用窗口末价与窗口 VWAP 聚合）。
    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。"""),
}

if __name__ == "__main__":
    for path, (fn, extras, weights, text) in SPECS.items():
        s = soft_peer(BASE, power=2.0)
        s = fn(s)
        s = keep(s, extras)
        s = tail(s, extras, weights)
        s = doc(s, text)
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
