# S6：同一次训练的不同 epoch 断点怎么用最好。对每个 run 生成三个变体，写到 e2e/runs/S6/<run>_<变体>/，再用 cmp_runs.py 两折对比：
#   ep2   = 第 2 个 epoch 的权重
#   swa23 = 第 2、3 个 epoch 权重逐参数平均（单模型，平台推理成本不变）
#   ens23 = 第 2、3 个 epoch 的预测逐截面 z 分数后平均（两次前向）
# ep3 就是 run 本身（val_pred.parquet / pred_2024.parquet）。
# 用法：python e2e/scripts/s6_epoch.py e2e/runs/H4_winfeat7 [e2e/runs/S5a_wf7_icir ...]
import os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend

YEARS = (("2023", "val_pred.parquet"), ("2024", "pred_2024.parquet"))
dev = T.pick_device()
fb = FeatherBackend()


def predict(models, stats, year):
    parts = []
    for lo, hi in T.month_chunks(f"{year}-01-01 00:00:00", f"{year}-12-31 23:59:59"):
        parts.append(T.predict_range(models, fb, str(lo), str(hi), fb.pool(str(lo), str(hi)), stats, dev,
                                     out_neutral=0, keep_chars=True, zero=False))
    return pd.concat(parts, ignore_index=True)


def cs_z(d):
    g = d.groupby("date")["factor"]
    return (d["factor"] - g.transform("mean")) / (g.transform("std") + 1e-12)


for run in sys.argv[1:]:
    name = os.path.basename(run.rstrip("/"))
    ck = T.load_model(os.path.join(run, "transformer_model.json"))
    T.configure_from_ckpt(ck)
    T.LABEL_NEUTRAL = max(T.LABEL_NEUTRAL, 1e-9)   # 让 need_chars() 为真，以便输出风格特征列
    stats = T.load_stats(ck)
    sd2 = torch.load(os.path.join(run, "ckpt", "m0_ep2.pt"), map_location="cpu")["model"]
    sd3 = torch.load(os.path.join(run, "ckpt", "m0_ep3.pt"), map_location="cpu")["model"]
    variants = {"ep2": sd2, "swa23": {k: (sd2[k].float() + sd3[k].float()) / 2 if sd2[k].is_floating_point() else sd3[k]
                                      for k in sd3}}
    for vn, sd in variants.items():
        out = os.path.join("e2e/runs/S6", f"{name}_{vn}")
        os.makedirs(out, exist_ok=True)
        m = T.build_models(dict(ck, state_dicts=[sd]), dev)
        for year, fn in YEARS:
            if os.path.exists(os.path.join(out, fn)):
                continue
            predict(m, stats, year).to_parquet(os.path.join(out, fn))
            print(name, vn, year, "done", flush=True)
    out = os.path.join("e2e/runs/S6", f"{name}_ens23")
    os.makedirs(out, exist_ok=True)
    for year, fn in YEARS:
        a = pd.read_parquet(os.path.join("e2e/runs/S6", f"{name}_ep2", fn))
        b = pd.read_parquet(os.path.join(run, fn))
        a = a.assign(factor=cs_z(a))
        b = b[["date", "instrument", "factor"]].assign(factor=cs_z(b)).rename(columns={"factor": "f3"})
        m_ = a.merge(b, on=["date", "instrument"], how="inner")
        m_["factor"] = (m_["factor"] + m_["f3"]) / 2
        m_.drop(columns="f3").to_parquet(os.path.join(out, fn))
        print(name, "ens23", year, len(a), len(b), len(m_), flush=True)
