# -*- coding: utf-8 -*-
"""CPU 推理测速（模拟平台 8 核机器）：用指定 JSON 对一个月做 predict_range，分别计时切窗与前向。
用法：python e2e/scripts/infer_bench.py <json> <threads> [YYYY-MM]
"""
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "submission"))
import Transformer_modelsave_train as T  # noqa: E402
from feather_data import FeatherBackend, set_read_cols  # noqa: E402

path, threads = sys.argv[1], int(sys.argv[2])
month = sys.argv[3] if len(sys.argv) > 3 else "2024-03"
torch.set_num_threads(threads)
ck = T.load_model(path)
T.configure_from_ckpt(ck)
set_read_cols(T.FEATURE_COLS)
dev = torch.device("cpu")
models = T.build_models(ck, dev)
stats = T.load_stats(ck)
fb = FeatherBackend()
lo, hi = f"{month}-01 00:00:00", (T.pd.Timestamp(f"{month}-01") + T.pd.offsets.MonthEnd(0)).strftime("%Y-%m-%d 23:59:59")
inst = fb.pool(lo, hi)
if ck.get("arch") == "e04":  # E04：分别计时 数据与特征 / 前向（e04_predict_array 会把线程数设为 os.cpu_count，这里固定为 threads）
    os.cpu_count = lambda: threads
    t0 = time.time()
    X, _, keys, _ = T.build_dataset(fb, lo, hi, "predict", inst, stats)
    t1 = time.time()
    T.predict_any(models, X, keys, dev)
    t2 = time.time()
    n = len(keys)
    print(f"[e04] threads={torch.get_num_threads()} members={len(models)} samples={n:,} 数据与特征 {t1 - t0:.1f}s 前向 {t2 - t1:.1f}s "
          f"({n / (t2 - t1):,.0f} 样本/秒)；按每月 18 万样本估算 1 年约 {(t2 - t0) / n * 180000 * 12 / 60:.0f} 分钟（不含平台读数据）")
    sys.exit(0)
t0 = time.time()
chunks = list(T.iter_chunks(fb, lo, hi, inst, with_label=False, norm=ck["norm"]))
t1 = time.time()
n = 0
for Xf, Xc, Xs, _, keys in chunks:
    X = (T.standardize(Xf, stats[0], stats[1]), None, Xs)
    T.predict_array(models, X, dev)
    n += len(keys)
t2 = time.time()
print(f"threads={threads} members={len(models)} samples={n:,} 切窗+特征 {t1 - t0:.1f}s 前向 {t2 - t1:.1f}s "
      f"({n / (t2 - t1):,.0f} 样本/秒)；按每月 18 万样本估算 1 年约 {(t2 - t0) / n * 180000 * 12 / 60:.0f} 分钟（不含平台读数据）")
