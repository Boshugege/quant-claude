# 10-25：E17（E16 + 树成员）与 E16 在 2024-03 的预测对比：截面秩相关、缺失、09:30。
import json, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend
fb = FeatherBackend(); lo, hi = "2024-03-01 00:00:00", "2024-03-31 23:59:59"; dev = torch.device("cpu")
out = {}
for nm, p in (("E16", "e2e/提交/E16_秩目标六轮_多任务头0.5_两成员加中间时刻_2020-24_4种子集成/transformer_model.json"), ("E17", "e2e/提交/E17_E16四成员加树模型成员_2020-24/transformer_model.json")):
    ck = T.load_model(p, map_location="cpu"); T.configure_from_ckpt(ck)
    out[nm] = T.e04_predict_range(T.build_models(ck, dev), fb, lo, hi, fb.pool(lo, hi), T.load_stats(ck), dev)
    print(nm, "树成员", bool(T.GBDT_CK), "行", len(out[nm]), "nan", int(out[nm]["factor"].isna().sum()), flush=True)
m = out["E16"].merge(out["E17"], on=["date", "instrument"]); m = m[m["date"].dt.strftime("%H%M") != "0930"]
c = m.groupby("date")[["factor_x", "factor_y"]].corr(method="spearman").xs("factor_x", level=1)["factor_y"]
print("E16 与 E17 截面秩相关 均值 %.4f 最小 %.4f" % (c.mean(), c.min()))
