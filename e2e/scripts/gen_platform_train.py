# -*- coding: utf-8 -*-
"""把某次 fold-A 实验的配置改写成"全量重训用"的训练脚本（在 A100 官方环境上直接跑 train_and_save）。

和 package.py 用同一套常量改写规则，区别：
- 不需要训练好的权重：结构超参取自该实验的 transformer_model.json（只读 model_cfg 等配置字段）；
- 训练区间、种子个数、epoch 可以覆盖（默认 2020-01-01 ~ 2024-12-31、4 个种子、沿用实验的 epoch）；
- 输出到 e2e/platform/<名称>/Transformer_modelsave_train.py，连同 a100_probe.py / a100_full.py 一起上传。

用法（macstudio 上，实验目录在那边）：
  python e2e/scripts/gen_platform_train.py e2e/runs/H4_winfeat7 E08_H4 [--n-models 4] [--epochs 3] [--icir-w 0.5]
"""
import argparse
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.normpath(os.path.join(HERE, ".."))
ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("name")
ap.add_argument("--n-models", type=int, default=4)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--train", nargs=2, default=["2020-01-01", "2024-12-31"])
ap.add_argument("--out-neutral", type=float, default=None)
ap.add_argument("--icir-w", type=float, default=None)
ap.add_argument("--turn-w", type=float, default=None)
ap.add_argument("--day-batch", type=int, default=None)
a = ap.parse_args()

cfg = json.load(open(os.path.join(a.run_dir, "config.json")))
ck = json.load(open(os.path.join(a.run_dir, "transformer_model.json")))
src = open(os.path.join(E2E, "submission", "Transformer_modelsave_train.py"), encoding="utf-8").read()
assert ck.get("arch") == "e04", "只支持 e04 结构"


def setc(text, name_, value):
    pat = re.compile(rf"^({name_}\s*=\s*)([^#\n]*?)(\s*#.*)?$", re.M)
    assert pat.search(text), name_
    return pat.sub(lambda m: m.group(1) + value + (("  " + m.group(3).strip()) if m.group(3) else ""), text, count=1)


tr0, tr1 = a.train
src = re.sub(r'^TRAIN_START, TRAIN_END = .*$',
             f'TRAIN_START, TRAIN_END = "{tr0}", "{tr1} 23:59:59"   # 训练区间写死，切勿用测试区间训练', src, count=1, flags=re.M)
mc = ck["model_cfg"]
src = setc(src, "ARCH", json.dumps("e04"))
for k, v in [("SEQ_LEN", mc["seq_len"]), ("COARSE_LEN", mc.get("coarse_len", 0)),
             ("SUMMARY", bool(mc.get("n_summary", 0))), ("WEIGHT_DECAY", cfg["wd"]),
             ("LR_SCHEDULE", json.dumps(cfg["sched"])), ("N_MODELS", a.n_models),
             ("NORM", json.dumps(ck["norm"])), ("TARGET", json.dumps(ck["target"])),
             ("LOSS", json.dumps(ck.get("loss", "mse"))), ("SECS_PER_BATCH", cfg.get("secs_per_batch", 4)),
             ("FIELD_SET", json.dumps(ck["field_set"])), ("MAX_TRAIN_INSTRUMENTS", cfg["max_inst"]),
             ("SUMMARY_SET", json.dumps(ck.get("summary_set", "all"))),
             ("SUMMARY_POOL", bool(mc.get("summary_pool", False))),
             ("LABEL_NEUTRAL", float(ck.get("label_neutral", 0.0)))]:
    src = setc(src, k, str(v))
src = re.sub(r"^EPOCHS, BATCH, LR, SEED = .*$",
             f"EPOCHS, BATCH, LR, SEED = {a.epochs or cfg['epochs']}, {cfg['batch']}, {cfg['lr']}, {cfg['seed']}",
             src, count=1, flags=re.M)
e04 = {k: v for k, v in mc.items() if k not in ("n_feat", "seq_len")}
src = re.sub(r"^E04_CFG = dict\(.*?\)\n", "E04_CFG = dict(" + ", ".join(f"{k}={v!r}" for k, v in e04.items()) + ")\n",
             src, count=1, flags=re.M | re.S)
src = setc(src, "WIN_LOOKBACK_DAYS", str(ck.get("win_lookback_days", 130)))
src = setc(src, "TRAIN_NEUTRAL", json.dumps(ck.get("train_neutral", "none")))
src = setc(src, "WIN_GROUPS", json.dumps(ck.get("win_groups", "shape" if mc.get("n_win_feat") == 7 else "")))
over = {"ICIR_WEIGHT": a.icir_w, "TURN_WEIGHT": a.turn_w, "DAY_BATCH": a.day_batch}
for k_, v_ in [("PIND_K", ck.get("pind_k", 30)), ("PIND_DAYS", ck.get("pind_days", 120)),
               ("ZERO_0930", bool(ck.get("zero_0930", False))), ("LOSS_SKIP_0930", bool(ck.get("loss_skip_0930", False))),
               ("DAY_BATCH", int(ck.get("day_batch", 0))), ("TURN_WEIGHT", float(ck.get("turn_weight", 0.0))),
               ("ICIR_WEIGHT", float(ck.get("icir_weight", 0.0))),
               ("TARGET_AUX_W", float(ck.get("target_aux_w", 0.0))), ("TARGET_AUX_N", int(ck.get("target_aux_n", 2))), ("TARGET_AUX_DECAY", float(ck.get("target_aux_decay", 1.0))),
               ("PEER_FROM_DAILY", bool(ck.get("peer_from_daily", False)))]:
    if over.get(k_) is not None:
        v_ = type(v_)(over[k_])
    src = setc(src, k_, str(v_))
src = setc(src, "OUT_NEUTRAL", str(float(a.out_neutral if a.out_neutral is not None else ck.get("out_neutral", 1.0))))
assert "cpt_jyc_2026_exposure" not in src, "提交代码不得读取官方风格暴露表"
assert "CKPT_DIR = None" in src

out = os.path.join(E2E, "platform", a.name)
os.makedirs(out, exist_ok=True)
open(os.path.join(out, "Transformer_modelsave_train.py"), "w", encoding="utf-8").write(src)
for f in ("a100_probe.py", "a100_full.py"):
    shutil.copy(os.path.join(E2E, "platform", f), os.path.join(out, f))
keys = ["TRAIN_START, TRAIN_END", "N_MODELS", "EPOCHS, BATCH, LR, SEED", "E04_CFG", "TRAIN_NEUTRAL", "ZERO_0930",
        "LOSS_SKIP_0930", "DAY_BATCH", "TURN_WEIGHT", "ICIR_WEIGHT", "PEER_FROM_DAILY", "OUT_NEUTRAL"]
for line in src.splitlines():
    if any(line.startswith(k + " =") for k in keys):
        print("  " + line.split("  #")[0])
print("written", out)
