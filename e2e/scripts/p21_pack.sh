#!/bin/bash
# P21／P22（10-26 晚，探针）：fold-A 单模型 + fold-A 树成员（G8 配置，2020–22 训练，权重 0.25 = 五分之一，4090 上 e17_add_gbdt 训出）。
#   P22 = R37a（腿换手 0.1，即 P17 的模型）+ 树：公榜直接对 P17 看"腿换手 + 树"是否叠加；
#   P21 = R41a（腿换手 0.15，4090）+ 树：对 P22 看 0.15 是否比 0.1 好（P18 的 0.3 崩盘，悬崖在 0.1～0.3 之间）。
set -e
cd "$(dirname "$0")/../.."
mkdir -p e2e/runs/R41a_leg015 e2e/runs/fA_tree
scp -q "4090:D:/quant-claude/e2e/runs/R41a_leg015/transformer_model.json" "4090:D:/quant-claude/e2e/runs/R41a_leg015/config.json" e2e/runs/R41a_leg015/
scp -q "4090:D:/quant-claude/e2e/runs/R37b_leg03_g8/transformer_model.json" e2e/runs/fA_tree/
for pair in "R37a_leg01:R37a_leg01_g8:P22_探针_P17加树模型成员_2020-22" "R41a_leg015:R41a_leg015_g8:P21_探针_腿换手0.15加树模型成员_2020-22"; do
    src=${pair%%:*}; rest=${pair#*:}; dst=${rest%%:*}; N=${rest#*:}
    mkdir -p e2e/runs/$dst; cp e2e/runs/$src/{transformer_model.json,config.json} e2e/runs/$dst/
    python3 - e2e/runs/$dst e2e/runs/fA_tree <<'PY'
import json, os, sys
d, g = sys.argv[1], sys.argv[2]
ck = json.load(open(os.path.join(d, "transformer_model.json")))
gb = json.load(open(os.path.join(g, "transformer_model.json")))["gbdt"]
assert gb["cfg"]["weight"] == 0.25 and len(ck["state_dicts"]) == 1, "单模型 + 树权重 0.25"
ck["gbdt"] = gb
json.dump(ck, open(os.path.join(d, "transformer_model.json"), "w"))
c = json.load(open(os.path.join(d, "config.json"))); c["gbdt"] = gb["cfg"]
json.dump(c, open(os.path.join(d, "config.json"), "w"), ensure_ascii=False, indent=1)
print(d, "接上树", gb["cfg"]["weight"], "成员数", len(ck["state_dicts"]))
PY
    python3 e2e/scripts/package.py e2e/runs/$dst "$N" 2>&1 | tail -1
    grep -n "^E04_LEG_TURN\|^E04_GBDT\|^TRAIN_START" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-150
    rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
    bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert|树" | tail -4
done
