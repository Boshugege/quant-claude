# -*- coding: utf-8 -*-
"""打包交付：把某次远端训练产物整理成 e2e/提交/E<NN>_<简述>/ 三个文件 + zip（本机运行）。

训练脚本里的配置常量会改写成与 JSON 完全一致的值（私榜重训调用 train_and_save 时读取这些常量），
然后逐项核对；推理 notebook 只保留说明与 main 两个单元。
用法：python3 e2e/scripts/package.py <run_dir 含 transformer_model.json 与 config.json> E02_简述
"""
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
E2E = os.path.normpath(os.path.join(HERE, ".."))
run_dir, name = sys.argv[1], sys.argv[2]
# 可选第三个参数：覆盖推理输出中性化力度（同一次训练出 E07 / E07b 探针两版时用；同时改写 JSON 里的 out_neutral）
OUT_NEUTRAL_OVERRIDE = float(sys.argv[3]) if len(sys.argv) > 3 else None
# 环境变量 DAILY_MIX：覆盖推理时混入日线专家头的权重（S23；同时写进 JSON 的 daily_mix，训练脚本常量 E04_DAILY_MIX 同步）
DAILY_MIX = float(os.environ["DAILY_MIX"]) if os.environ.get("DAILY_MIX") else None
out = os.path.join(E2E, "提交", name)
if os.path.exists(out):
    sys.exit(f"{out} 已存在：已交付版本不可覆盖，请换编号")

cfg = json.load(open(os.path.join(run_dir, "config.json")))
ck = json.load(open(os.path.join(run_dir, "transformer_model.json")))
src = open(os.path.join(E2E, "submission", "Transformer_modelsave_train.py"), encoding="utf-8").read()


def setc(text, name_, value):
    pat = re.compile(rf"^({name_}\s*=\s*)([^#\n]*?)(\s*#.*)?$", re.M)
    assert pat.search(text), name_
    return pat.sub(lambda m: m.group(1) + value + (("  " + m.group(3).strip()) if m.group(3) else ""), text, count=1)


arch = ck.get("arch", "v1")
tr = cfg["train"]
tr_end = tr[1] if len(tr[1]) > 10 else tr[1] + " 23:59:59"
mc = ck["model_cfg"]
src = re.sub(r'^TRAIN_START, TRAIN_END = .*$',
             f'TRAIN_START, TRAIN_END = "{tr[0]}", "{tr_end}"   # 训练区间写死，切勿用测试区间训练', src, count=1, flags=re.M)
src = setc(src, "ARCH", json.dumps(arch))
for k, v in [("SEQ_LEN", mc["seq_len"]), ("COARSE_LEN", mc.get("coarse_len", 0)),
             ("SUMMARY", bool(mc.get("n_summary", 0))), ("WEIGHT_DECAY", cfg["wd"]),
             ("LR_SCHEDULE", json.dumps(cfg["sched"])), ("N_MODELS", len(ck["state_dicts"])),
             ("NORM", json.dumps(ck["norm"])), ("E04_NORM", json.dumps(ck["norm"] if ck.get("norm") in ("rel2", "rel3") else "rel2")), ("E04_GBDT", repr(ck["gbdt"]["cfg"]) if ck.get("gbdt") else "{}"), ("PAIR_UNITS", repr(bool(ck.get("pair_units", False)))), ("E04_LEG_TURN", repr(float(ck.get("leg_turn", 0.0)))), ("E04_LEG_TAU", repr(float(ck.get("leg_tau", 0.15)))), ("TARGET", json.dumps(ck["target"])),
             ("LOSS", json.dumps(ck.get("loss", "mse"))), ("SECS_PER_BATCH", cfg.get("secs_per_batch", 4)),
             ("FIELD_SET", json.dumps(ck["field_set"])), ("MAX_TRAIN_INSTRUMENTS", cfg["max_inst"]),
             ("SUMMARY_SET", json.dumps(ck.get("summary_set", "all"))),
             ("SUMMARY_POOL", bool(mc.get("summary_pool", False))),
             ("LABEL_NEUTRAL", float(ck.get("label_neutral", 0.0))), ("OUT_NEUTRAL", float(ck.get("out_neutral", 0.0)))]:
    src = setc(src, k, str(v))
src = re.sub(r"^EPOCHS, BATCH, LR, SEED = .*$", f"EPOCHS, BATCH, LR, SEED = {cfg['epochs']}, {cfg['batch']}, {cfg['lr']}, {cfg['seed']}",
             src, count=1, flags=re.M)
if arch == "v1":
    src = re.sub(r"^MODEL_CFG = dict\(n_feat=N_FEAT, d_model=\d+, nhead=\d+, nlayers=\d+, dim_ff=\d+,",
                 f"MODEL_CFG = dict(n_feat=N_FEAT, d_model={mc['d_model']}, nhead={mc['nhead']}, nlayers={mc['nlayers']}, dim_ff={mc['dim_ff']},",
                 src, count=1, flags=re.M)
else:
    e04 = {k: v for k, v in mc.items() if k not in ("n_feat", "seq_len")}
    src = re.sub(r"^E04_CFG = dict\(.*?\)\n", "E04_CFG = dict(" + ", ".join(f"{k}={v!r}" for k, v in e04.items()) + ")\n",
                 src, count=1, flags=re.M | re.S)
    src = setc(src, "WIN_LOOKBACK_DAYS", str(ck.get("win_lookback_days", 130)))
    src = setc(src, "TRAIN_NEUTRAL", json.dumps(ck.get("train_neutral", "none")))
    src = setc(src, "WIN_GROUPS", json.dumps(ck.get("win_groups", "shape" if mc.get("n_win_feat") == 7 else "")))
    for k_, v_ in [("PIND_K", ck.get("pind_k", 30)), ("PIND_DAYS", ck.get("pind_days", 120)),
                   ("ZERO_0930", bool(ck.get("zero_0930", False))), ("LOSS_SKIP_0930", bool(ck.get("loss_skip_0930", False))),
                   ("DAY_BATCH", int(ck.get("day_batch", 0))), ("TURN_WEIGHT", float(ck.get("turn_weight", 0.0))),
                   ("ICIR_WEIGHT", float(ck.get("icir_weight", 0.0))),
                   ("TARGET_AUX_W", float(ck.get("target_aux_w", 0.0))), ("TARGET_AUX_N", int(ck.get("target_aux_n", 2))), ("TARGET_AUX_DECAY", float(ck.get("target_aux_decay", 1.0))),
                   ("PEER_FROM_DAILY", bool(ck.get("peer_from_daily", False))), ("E04_HEAD_W", float(ck.get("head_w", 1.0))),
                   ("YEAR_W", repr({int(k): float(v) for k, v in ck.get("year_w", {}).items()})),
                   ("SLOT_W", repr({str(k): float(v) for k, v in ck.get("slot_w", {}).items()})),
                   ("EXTRA_TIMES", repr(list(ck.get("extra_times", [])))),
                   ("E04_EXTRA_MEMBERS", repr(list(ck.get("extra_members", [])))),
                   ("E04_GRU_MEMBERS", repr(list(ck.get("gru_members", [])))),
                   ("E04_WEIGHT_EMA", str(float(ck.get("weight_ema", 0.0)))),
                   ("E04_DAILY_MIX", str(float(DAILY_MIX if DAILY_MIX is not None else ck.get("daily_mix", 0.0)))),
                   ("E04_ANCHOR_TRAIN", str(float(ck.get("anchor_train", 0.0)))),
                   ("E04_MTL_W", str(float(ck.get("mtl_w", 0.0)))),                  # 10-23：多任务头的训练常量（私榜重训用；推理不需要）
                   ("E04_MTL_SET", json.dumps(ck["mtl_set"] if isinstance(ck.get("mtl_set"), str) else ",".join(ck.get("mtl_set") or ["rv", "lv"])))]:
        src = setc(src, k_, str(v_))
    src = setc(src, "OUT_NEUTRAL", str(float(OUT_NEUTRAL_OVERRIDE if OUT_NEUTRAL_OVERRIDE is not None else ck.get("out_neutral", 0.0))))
assert "cpt_jyc_2026_exposure" not in src, "E07 起提交代码不得读取官方风格暴露表"
assert "CKPT_DIR = None" in src

os.makedirs(out)
open(os.path.join(out, "Transformer_modelsave_train.py"), "w", encoding="utf-8").write(src)
if OUT_NEUTRAL_OVERRIDE is None and DAILY_MIX is None:
    shutil.copy(os.path.join(run_dir, "transformer_model.json"), os.path.join(out, "transformer_model.json"))
else:
    if OUT_NEUTRAL_OVERRIDE is not None:
        ck["out_neutral"] = OUT_NEUTRAL_OVERRIDE
    if DAILY_MIX is not None:
        ck["daily_mix"] = DAILY_MIX
    json.dump(ck, open(os.path.join(out, "transformer_model.json"), "w", encoding="utf-8"), ensure_ascii=False)
nb = json.load(open(os.path.join(E2E, "submission", "Transformer_modelsave_predict.ipynb"), encoding="utf-8"))
nb["cells"] = nb["cells"][:2]
json.dump(nb, open(os.path.join(out, "Transformer_modelsave_predict.ipynb"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
with zipfile.ZipFile(os.path.join(E2E, "提交", name + ".zip"), "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("Transformer_modelsave_train.py", "Transformer_modelsave_predict.ipynb", "transformer_model.json"):
        z.write(os.path.join(out, f), f)
print("packaged", out, {f: os.path.getsize(os.path.join(out, f)) for f in os.listdir(out)})
print("verify on remote with: python e2e/scripts/verify_package.py", name)
subprocess.run(["python3", "-m", "py_compile", os.path.join(out, "Transformer_modelsave_train.py")], check=True)
