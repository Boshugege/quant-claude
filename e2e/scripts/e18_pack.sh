#!/bin/bash
# E18 打包：取回四个成员（F40–F43，E16 配方 + 腿换手损失 0.1，种子 42–45）→ 合并 → 两个包：
#   E18  只有四个 Transformer 成员；
#   E18g 再接上 E17b 的 2020–24 树成员（G8 配置，与 Transformer 成员无关；fold-A 上 L1 + 树：IC +0.003、多空 +1.9、换手 +0.013）。
# 同步到 macstudio，两个都 verify_package。本机项目根目录运行。
set -e
cd "$(dirname "$0")/../.."
RUNS="F40_2020-24_E18_leg01_s42 F41_2020-24_E18_leg01_s43 F42_2020-24_E18_leg01_s44 F43_2020-24_E18_leg01_s45"
for r in $RUNS; do
    mkdir -p e2e/runs/$r
    rsync -a macstudio:Desktop/work/quant/e2e/runs/$r/{transformer_model.json,config.json,metrics.json,log.txt} e2e/runs/$r/
done
M=e2e/runs/F40-43_2020-24_E18_leg01_4seeds
python3 e2e/scripts/merge_members.py $M $(for r in $RUNS; do echo e2e/runs/$r; done)
G=e2e/runs/F30-33G8_2020-24_E17b_gbdt_win12
MG=${M}_gbdt
mkdir -p $MG; cp $M/* $MG/ 2>/dev/null || true
python3 - "$MG" "$G" <<'PY'
import json, os, sys
mg, g = sys.argv[1], sys.argv[2]
ck = json.load(open(os.path.join(mg, "transformer_model.json")))
ck["gbdt"] = json.load(open(os.path.join(g, "transformer_model.json")))["gbdt"]
json.dump(ck, open(os.path.join(mg, "transformer_model.json"), "w"))
c = json.load(open(os.path.join(mg, "config.json"))); c["gbdt"] = ck["gbdt"]["cfg"]
json.dump(c, open(os.path.join(mg, "config.json"), "w"), ensure_ascii=False, indent=1)
print("接上树成员", ck["gbdt"]["cfg"])
PY
for pair in "$M:E18_E16配方加腿换手损失0.1_2020-24_4种子集成" "$MG:E18g_E18加树模型成员_2020-24"; do
    run=${pair%%:*}; N=${pair#*:}
    python3 e2e/scripts/package.py $run "$N" 2>&1 | tail -1
    grep -n "^N_MODELS\|^EPOCHS, BATCH, LR, SEED\|^PAIR_UNITS\|^E04_LEG_TURN\|^E04_GBDT\|^E04_EXTRA" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-160
    rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
    bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert|树" | tail -4
done
