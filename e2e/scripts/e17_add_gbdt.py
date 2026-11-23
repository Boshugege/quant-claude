# 10-25：E17 = E16 的四个 Transformer 成员（不重训）+ 2020–24 全量训练的树模型成员。
# 用法：python e2e/scripts/e17_add_gbdt.py <E16 合并 run 目录> <输出 run 目录> '<E04_GBDT json>'
import json, os, shutil, sys, time
import torch
sys.path.insert(0, "e2e/src"); sys.path.insert(0, "e2e/submission")
import Transformer_modelsave_train as T
from feather_data import FeatherBackend
src, out = sys.argv[1], sys.argv[2]
cfg = json.load(open(sys.argv[3])) if sys.argv[3].endswith(".json") else json.loads(sys.argv[3])
ck = json.load(open(os.path.join(src, "transformer_model.json")))
T.TARGET, T.TARGET_AUX_W, T.TARGET_AUX_N, T.TARGET_AUX_DECAY = ck["target"], ck["target_aux_w"], ck["target_aux_n"], ck.get("target_aux_decay", 1.0)
T.TRAIN_NEUTRAL, T.PIND_K, T.PIND_DAYS = ck["train_neutral"], ck.get("pind_k", T.PIND_K), ck.get("pind_days", T.PIND_DAYS)
T.TRAIN_START, T.TRAIN_END = ck["train_range"]
T.WIN_GROUPS, T.WIN_LOOKBACK_DAYS = ck["win_groups"], ck["win_lookback_days"]
T.E04_CFG.update({k: v for k, v in ck["model_cfg"].items() if k in T.E04_CFG})
T.E04_GBDT = cfg
dev = T.pick_device()
fb = FeatherBackend()
inst = fb.pool(T.TRAIN_START, T.TRAIN_END)
t0 = time.time()
gb = T.gbdt_train(fb, inst, dev)
print("树模型成员训练完成 %.0fs" % (time.time() - t0), flush=True)
os.makedirs(out, exist_ok=True)
for f in os.listdir(src):
    if f != "transformer_model.json" and os.path.isfile(os.path.join(src, f)):
        shutil.copy(os.path.join(src, f), out)
ck["gbdt"] = gb
json.dump(ck, open(os.path.join(out, "transformer_model.json"), "w"))
cfgf = os.path.join(out, "config.json")
if os.path.exists(cfgf):
    c = json.load(open(cfgf)); c["gbdt"] = cfg; json.dump(c, open(cfgf, "w"), ensure_ascii=False, indent=1)
print("写出", out, os.path.getsize(os.path.join(out, "transformer_model.json")), "bytes")
