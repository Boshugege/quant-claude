# -*- coding: utf-8 -*-
"""A100 官方环境上全量重训（就是私榜会做的那一步）。与 Transformer_modelsave_train.py 放在同一目录运行，建议后台：
    nohup python a100_full.py > train_log.txt 2>&1 &
    tail -f train_log.txt
训练配置全部来自同目录的 Transformer_modelsave_train.py（区间、种子数、epoch 都写死在里面），本脚本不改任何配置，
只额外打开断点目录 ckpt/（每个种子每个 epoch 存一次，会话断了重跑本脚本会从断点续训）。
完成后下载 transformer_model.json 和 train_log.txt 交回即可。"""
import os, sys, time, json, resource
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import Transformer_modelsave_train as T

T.CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ckpt")
t0 = time.time()
print(f"[full] 开始 {time.strftime('%F %T')}  训练区间 {T.TRAIN_START} ~ {T.TRAIN_END}  种子 {T.N_MODELS}  epoch {T.EPOCHS}  "
      f"设备 {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}", flush=True)
p = T.train_and_save({"bar1m": "cpt_jyc_2026_stock_bar1m"}, model_path="transformer_model.json")
ck = json.load(open(p))
print(f"[full] 完成 {time.strftime('%F %T')}  用时 {(time.time() - t0) / 60:.1f} 分钟  模型 {os.path.getsize(p) / 2**20:.1f} MiB  "
      f"成员 {len(ck['state_dicts'])}  内存峰值 {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20:.1f}GB"
      + (f"  显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f}GB" if torch.cuda.is_available() else ""), flush=True)
