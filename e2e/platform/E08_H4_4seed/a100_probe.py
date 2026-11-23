# -*- coding: utf-8 -*-
"""A100 官方环境探查 + 小规模测速（约 10–20 分钟）。与 Transformer_modelsave_train.py 放在同一目录运行：
    python a100_probe.py            （或在 notebook 里 %run a100_probe.py）
输出：环境信息、dai 取数耗时、一个月数据 1 个 epoch 的完整 train_and_save（数据构建耗时 + 训练吞吐 + 显存/内存峰值）。
把全部输出复制回来即可。不写任何正式文件（模型存为 probe_model.json，可删）。"""
import os, sys, time, platform, resource, shutil, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()


def stamp(msg):
    print(f"[probe {time.time() - T0:7.1f}s] {msg}", flush=True)


import torch
stamp(f"python {platform.python_version()}  torch {torch.__version__}  cuda {torch.cuda.is_available()}  "
      f"gpu {torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}  "
      f"gpu_mem {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f}GB" if torch.cuda.is_available() else "")
stamp(f"cpu {os.cpu_count()}  disk_free {shutil.disk_usage('.').free / 2**30:.0f}GB")
try:
    import psutil
    stamp(f"ram total {psutil.virtual_memory().total / 2**30:.0f}GB  avail {psutil.virtual_memory().available / 2**30:.0f}GB")
except Exception as e:
    stamp(f"psutil 不可用：{e}")
for m in ("numpy", "pandas", "duckdb", "pyarrow", "structlog"):
    try:
        stamp(f"{m} {__import__(m).__version__}")
    except Exception as e:
        stamp(f"{m} 不可用：{e}")

import dai
t = time.time()
df = dai.query("SELECT date, instrument, close FROM cpt_jyc_2026_stock_bar1m",
               filters={"date": ["2024-01-02 00:00:00", "2024-01-02 23:59:59"]}).df()
stamp(f"dai 取一天 1m 数据：{len(df):,} 行，{time.time() - t:.1f}s；date 范围 {df['date'].min()} ~ {df['date'].max()}")
t = time.time()
df = dai.query("SELECT min(date) AS lo, max(date) AS hi FROM cpt_jyc_2026_stock_bar1m",
               filters={"date": ["2019-01-01 00:00:00", "2026-12-31 23:59:59"]}).df()
stamp(f"bar1m 可见区间：{df.iloc[0].to_dict()}（{time.time() - t:.1f}s）")

import Transformer_modelsave_train as T
T.TRAIN_START, T.TRAIN_END = "2024-01-01", "2024-01-31 23:59:59"
T.N_MODELS, T.EPOCHS = 1, 1
stamp(f"小规模 train_and_save：{T.TRAIN_START} ~ {T.TRAIN_END[:10]}，1 个种子 × 1 epoch（正式配置 N_MODELS/EPOCHS 不变，只在内存里改）")
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats()
t = time.time()
p = T.train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"}, model_path="probe_model.json")
el = time.time() - t
ck = json.load(open(p))
stamp(f"完成：{el:.0f}s；模型 {os.path.getsize(p) / 2**20:.1f} MiB，成员 {len(ck['state_dicts'])}")
stamp(f"进程内存峰值 {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f}GB"
      + (f"；显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if torch.cuda.is_available() else ""))
stamp("请把以上全部输出（含 分片完成 / 训练进度 日志行）复制回来")
