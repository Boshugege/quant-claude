#!/bin/bash
# E21 打包：E18 的四个 Transformer 成员（F40–F43，种子 42–45，macstudio）+ 两个 GRU 成员（F70、F71，种子 46、47，4090，同配方 --seq-enc gru）
# → 合并（GRU 成员按序号记进 gru_members，各成员结构存 member_cfgs）→ 接上 E17b 的树成员 → 打包 → verify_package（含平台路径极小规模重训，6 个成员）→ 截断检测。
set -e
cd "$(dirname "$0")/../.."
for r in F40_2020-24_E18_leg01_s42 F41_2020-24_E18_leg01_s43 F42_2020-24_E18_leg01_s44 F43_2020-24_E18_leg01_s45; do
    mkdir -p e2e/runs/$r
    rsync -a macstudio:Desktop/work/quant/e2e/runs/$r/{transformer_model.json,config.json,metrics.json,log.txt} e2e/runs/$r/
done
for r in F70_2020-24_E21_gru_leg01_s46 F71_2020-24_E21_gru_leg01_s47; do
    mkdir -p e2e/runs/$r
    for f in transformer_model.json config.json metrics.json log.txt; do scp -q "4090:D:/quant-claude/e2e/runs/$r/$f" e2e/runs/$r/; done
done
RUNS="F40_2020-24_E18_leg01_s42 F41_2020-24_E18_leg01_s43 F42_2020-24_E18_leg01_s44 F43_2020-24_E18_leg01_s45 F70_2020-24_E21_gru_leg01_s46 F71_2020-24_E21_gru_leg01_s47"
M=e2e/runs/F40-43_70-71_2020-24_E21_tf4_gru2
STATS_ATOL=${STATS_ATOL:-0.02} python3 e2e/scripts/merge_members.py $M $(for r in $RUNS; do echo e2e/runs/$r; done)
python3 - "$M" e2e/runs/F30-33G8_2020-24_E17b_gbdt_win12 <<'PY'
import json, os, sys
m, g = sys.argv[1], sys.argv[2]
ck = json.load(open(os.path.join(m, "transformer_model.json")))
ck["gbdt"] = json.load(open(os.path.join(g, "transformer_model.json")))["gbdt"]
json.dump(ck, open(os.path.join(m, "transformer_model.json"), "w"))
c = json.load(open(os.path.join(m, "config.json"))); c["gbdt"] = ck["gbdt"]["cfg"]
json.dump(c, open(os.path.join(m, "config.json"), "w"), ensure_ascii=False, indent=1)
print("接上树成员；成员数", len(ck["state_dicts"]), "gru_members", ck.get("gru_members"), "seq_enc", [x.get("seq_enc", "tf") for x in ck["member_cfgs"]])
PY
N="${1:-E21_E18g加两个GRU成员_2020-24}"
python3 e2e/scripts/package.py $M "$N" 2>&1 | tail -1
grep -n "^N_MODELS\|^EPOCHS, BATCH, LR, SEED\|^PAIR_UNITS\|^E04_LEG_TURN\|^E04_GBDT\|^E04_GRU_MEMBERS\|^EXTRA_TIMES" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-140
rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert|树|GRU" | tail -5
ssh macstudio "cd ~/Desktop/work/quant; /opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python e2e/scripts/leak_check.py 'e2e/提交/$N' 2024-03-04 '2024-03-06 10:30:00' '2024-03-06 14:30:00' 2>&1 | grep -E '截断|结论'" < /dev/null
