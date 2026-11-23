# -*- coding: utf-8 -*-
"""同伴强度探针：改变软同伴的 k（候选同伴数）与 power（相关度加权次方），用于定位最优边界。"""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from patch_platform import BASE, soft_peer, tail, doc  # noqa: E402

SPECS = {
    "因子实验/probe_peer400p4.py": (400, 4.0),
    "因子实验/probe_peer50p1.py": (50, 1.0),
}

if __name__ == "__main__":
    for path, (k, pw) in SPECS.items():
        s = soft_peer(BASE, power=pw)
        s = s.replace("PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = 100, 60, 30, %.1f" % pw,
                      "PEER_K, PEER_LOOKBACK, PEER_MIN_OBS, PEER_POWER = %d, 60, 30, %.1f" % (k, pw))
        s = tail(s, (), ())
        s = doc(s, "软同伴残差阶梯 + 四特征 1.5 倍中性化（同伴探针：k=%d、相关度 %s 次方加权，09:30 修复版）。\n\n"
                   "    用于定位软同伴强度的边界：114 号 k=100/2、128 号 k=200/3 的代理分依次为 0.823 / 0.834，\n"
                   "    本探针把两个旋钮继续外推/回退，确认最优区间是否在 k=200/3 附近。\n"
                   "    09:30 修复：09:30 窗口的末价取 09:25 集合竞价 bar 的价格。" % (k, pw))
        pathlib.Path(path).write_text(s, encoding="utf-8")
        print("写出", path, len(s))
