#!/bin/bash
# E20 打包：成员 0、1 = E18 的 F40（种子 42）、F41（43），成员 2、3 = F62（44）、F63（45，腿换手 + 中间时刻）→ 混合集成合并（统计量用成员 2）
# → 接上 E17b 的 2020–24 树成员 → 打包 → 同步 → verify_package → 截断检测。本机项目根目录运行。
set -e
cd "$(dirname "$0")/../.."
RUNS="F40_2020-24_E18_leg01_s42 F41_2020-24_E18_leg01_s43 F62_2020-24_E20_leg01_x7_s44 F63_2020-24_E20_leg01_x7_s45"
for r in $RUNS; do
    mkdir -p e2e/runs/$r
    rsync -a macstudio:Desktop/work/quant/e2e/runs/$r/{transformer_model.json,config.json,metrics.json,log.txt} e2e/runs/$r/
done
M=e2e/runs/F40-41-62-63_2020-24_E20_leg01_mixed_x7
MIXED_EXTRA=1 STATS_FROM=2 STATS_ATOL=${STATS_ATOL:-0.02} python3 e2e/scripts/merge_members.py $M $(for r in $RUNS; do echo e2e/runs/$r; done)
python3 - "$M" e2e/runs/F30-33G8_2020-24_E17b_gbdt_win12 <<'PY'
import json, os, sys
m, g = sys.argv[1], sys.argv[2]
ck = json.load(open(os.path.join(m, "transformer_model.json")))
ck["gbdt"] = json.load(open(os.path.join(g, "transformer_model.json")))["gbdt"]
json.dump(ck, open(os.path.join(m, "transformer_model.json"), "w"))
c = json.load(open(os.path.join(m, "config.json"))); c["gbdt"] = ck["gbdt"]["cfg"]
json.dump(c, open(os.path.join(m, "config.json"), "w"), ensure_ascii=False, indent=1)
print("接上树成员；成员数", len(ck["state_dicts"]), "extra_members", ck.get("extra_members"), "extra_times", ck.get("extra_times"))
PY
N="${1:-E20_E18g改混合集成_两成员加中间时刻配对修正_2020-24}"
python3 e2e/scripts/package.py $M "$N" 2>&1 | tail -1
grep -n "^N_MODELS\|^EPOCHS, BATCH, LR, SEED\|^PAIR_UNITS\|^E04_LEG_TURN\|^E04_GBDT\|^EXTRA_TIMES\|^E04_EXTRA_MEMBERS" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-140
rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert|树" | tail -4
ssh macstudio "cd ~/Desktop/work/quant; /opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python e2e/scripts/leak_check.py 'e2e/提交/$N' 2024-03-04 '2024-03-06 10:30:00' '2024-03-06 14:30:00' 2>&1 | grep -E '截断|结论'" < /dev/null
