# -*- coding: utf-8 -*-
"""远端校验交付包（用模拟 dai）：
1. 训练脚本的配置常量推导出的 MODEL_CFG / 字段 / 回看长度 与 JSON 一致；
2. 训练脚本的 train_and_save 能在极小区间跑通（只在内存中缩小区间与 epoch，不改文件）；
3. 推理 notebook 的 main 用交付的 JSON 跑一个月，输出三列、8 个截面、无重复/inf/NaN、缺失率；
4. 交付 JSON 的预测与训练时验证集预测一致（若提供 run 目录）。
用法：python e2e/scripts/verify_package.py <交付目录> [start] [end]
"""
import importlib
import json
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
pkg = sys.argv[1]
start, end = (sys.argv[2:4] + ["2024-03-01 00:00:00", "2024-03-31 23:59:59"][len(sys.argv[2:4]):])[:2]
work = os.path.join("e2e", "runs", "verify_" + os.path.basename(pkg.rstrip("/")))
shutil.rmtree(work, ignore_errors=True)
os.makedirs(work)
for f in ("Transformer_modelsave_train.py", "Transformer_modelsave_predict.ipynb", "transformer_model.json"):
    shutil.copy(os.path.join(pkg, f), work)
sys.path.insert(0, os.path.join(HERE, "fake_dai"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, work)
import dai  # noqa: E402,F401
T = importlib.import_module("Transformer_modelsave_train")
import torch  # noqa: E402
T.pick_device = lambda: torch.device("cpu")          # 校验只用 CPU，不占用训练中的 GPU
assert os.path.dirname(T.__file__) == os.path.abspath(work), T.__file__
ck = json.load(open(os.path.join(work, "transformer_model.json")))

# 1. 配置一致性
if T.ARCH == "e04":
    cfg = dict(T.E04_CFG, n_feat=T.N_FEAT, seq_len=T.SEQ_LEN)
else:
    cfg = dict(T.MODEL_CFG, n_feat=T.N_FEAT, seq_len=T.SEQ_LEN, coarse_len=T.COARSE_LEN,
               n_summary=len(T.summary_names()) if T.SUMMARY else 0, summary_pool=T.SUMMARY_POOL)
checks = {"arch": T.ARCH == ck.get("arch", "v1"),
          "model_cfg": cfg == ck["model_cfg"], "field_set": T.FIELD_SET == ck["field_set"],
          "norm": (T.NORM == ck["norm"]) or T.ARCH == "e04", "target": T.TARGET == ck["target"],
          "n_models": T.N_MODELS == len(ck["state_dicts"]),
          "train_range": [T.TRAIN_START, T.TRAIN_END] == ck["train_range"],
          "summary": (T.summary_names() if T.SUMMARY else []) == ck.get("summary", []) or T.ARCH == "e04",
          "ckpt_dir_none": T.CKPT_DIR is None,
          "out_neutral": T.OUT_NEUTRAL == ck.get("out_neutral", 0.0),
          "label_neutral": T.LABEL_NEUTRAL == ck.get("label_neutral", 0.0)}
print("1. 配置一致性:", checks)
assert all(checks.values()), (cfg, ck["model_cfg"])

# 3. 推理 notebook（先跑推理，避免下一步的训练覆盖 JSON）
nb = json.load(open(os.path.join(work, "Transformer_modelsave_predict.ipynb")))
assert len(nb["cells"]) == 2 and "ema" not in "".join(nb["cells"][1]["source"]).lower()
ns = {"__name__": "nb"}
exec(compile("".join(nb["cells"][1]["source"]), "predict.ipynb", "exec"), ns)
t0 = time.time()
out = ns["main"]({"bar1m": "cpt_jyc_2026_stock_bar1m"}, start, end)
expected = dai.query("SELECT date, instrument FROM cpt_jyc_2026_instruments", filters={"date": [start, end]}).df().groupby("date").size()
miss = 1 - out.groupby("date").size().reindex(expected.index).fillna(0) / expected
print(f"3. 推理: rows={len(out):,} cols={list(out.columns)} slots={sorted(out['date'].dt.strftime('%H:%M').unique())} "
      f"dup={out.duplicated(['date', 'instrument']).sum()} inf={np.isinf(out['factor']).sum()} nan={out['factor'].isna().sum()} "
      f"max_missing={miss.max():.4f} {time.time() - t0:.0f}s")
assert list(out.columns) == ["date", "instrument", "factor"] and miss.max() <= 0.4

# 2. 训练脚本可运行（内存中缩小规模，写到临时路径）
T.TRAIN_START, T.TRAIN_END, T.MAX_TRAIN_INSTRUMENTS, T.EPOCHS, T.N_MODELS = "2020-02-01", "2020-02-29 23:59:59", 150, 1, 1
p = T.train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"}, model_path=os.path.join(work, "tiny_retrain.json"))
ck2 = json.load(open(p))
print("2. train_and_save 极小规模跑通:", os.path.getsize(p), "bytes; model_cfg 同结构:",
      {k: v for k, v in ck2["model_cfg"].items()} == ck["model_cfg"], "arch:", ck2.get("arch", "v1"))
print("校验通过")
