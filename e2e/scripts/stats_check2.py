# 合并集成时成员改用第一个成员的标准化统计量的影响：同一成员用"自己的"与"第一个成员的"统计量在 2024-03 的预测相关。
# 用法：python e2e/scripts/stats_check2.py <第一个成员 run> <成员 run> [...]
import sys
import pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
import os
if os.environ.get("CPU") == "1":                                  # 10-24：训练占着 GPU 时只用 CPU
    import torch
    T.pick_device = lambda: torch.device("cpu")
from feather_data import FeatherBackend
ref = T.load_model(sys.argv[1] + "/transformer_model.json")
fb = FeatherBackend()
lo, hi = "2024-03-01 00:00:00", "2024-03-31 23:59:59"
for r in sys.argv[2:]:
    a = T.load_model(r + "/transformer_model.json")
    T.configure_from_ckpt(a)
    dev = T.pick_device()
    m = T.build_models(a, dev)
    p1 = T.predict_range(m, fb, lo, hi, fb.pool(lo, hi), T.load_stats(a), dev, out_neutral=0, zero=False)
    p2 = T.predict_range(m, fb, lo, hi, fb.pool(lo, hi), T.load_stats(ref), dev, out_neutral=0, zero=False)
    d = p1.merge(p2, on=["date", "instrument"])
    d = d[d["date"].dt.strftime("%H%M") != "0930"]
    c = d.groupby("date")[["factor_x", "factor_y"]].corr(method="spearman").xs("factor_x", level=1)["factor_y"]
    print(r.split("/")[-1], "自有 vs 第一个成员统计量：截面秩相关 均值", round(float(c.mean()), 6), "最小", round(float(c.min()), 6), flush=True)
