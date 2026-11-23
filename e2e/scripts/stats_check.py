# 合并集成时第二个成员改用第一个成员的标准化统计量：比较同一个模型用两套统计量在 2024-03 的预测（截面排名相关、最大排名差）。
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend
a = T.load_model("e2e/runs/F13_2020-24_fuse_aux05_seed43/transformer_model.json")
b = T.load_model("e2e/runs/F12_2020-24_fuse_aux05_1x3/transformer_model.json")
T.configure_from_ckpt(a)
dev = T.pick_device()
m, fb = T.build_models(a, dev), FeatherBackend()
lo, hi = "2024-03-01 00:00:00", "2024-03-31 23:59:59"
p1 = T.predict_range(m, fb, lo, hi, fb.pool(lo, hi), T.load_stats(a), dev, out_neutral=0, zero=False)
p2 = T.predict_range(m, fb, lo, hi, fb.pool(lo, hi), T.load_stats(b), dev, out_neutral=0, zero=False)
d = p1.merge(p2, on=["date", "instrument"])
d = d[d["date"].dt.strftime("%H%M") != "0930"]
c = d.groupby("date")[["factor_x", "factor_y"]].corr().xs("factor_x", level=1)["factor_y"]
print("截面排名相关 均值", round(float(c.mean()), 6), "最小", round(float(c.min()), 6), "最大排名差", round(float((d.factor_x - d.factor_y).abs().max()), 4))
