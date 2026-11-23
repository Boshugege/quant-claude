#!/bin/bash
# E16 打包：取回四个成员（F30–F33，E16 配方：秩目标 + 6 轮 + 多任务头 0.5；F32／F33 加中间时刻 7 个截面）→ 混合集成合并 → 打包 → 同步到 macstudio → verify_package。本机项目根目录运行。
set -e
cd "$(dirname "$0")/../.."
RUNS="F30_2020-24_E16_mtl05_ep6_s42 F31_2020-24_E16_mtl05_ep6_s43 F32_2020-24_E16_mtl05_ep6_x7_s44 F33_2020-24_E16_mtl05_ep6_x7_s45"
for r in $RUNS; do
    mkdir -p e2e/runs/$r
    rsync -a macstudio:Desktop/work/quant/e2e/runs/$r/{transformer_model.json,config.json,metrics.json,log.txt} e2e/runs/$r/
done
M=e2e/runs/F30-33_2020-24_E16_mtl05_ep6_mixed_x7_4seeds
MIXED_EXTRA=1 STATS_FROM=2 STATS_ATOL=${STATS_ATOL:-0.02} python3 e2e/scripts/merge_members.py $M $(for r in $RUNS; do echo e2e/runs/$r; done)
N="${1:-E16_秩目标六轮_多任务头0.5_两成员加中间时刻_2020-24_4种子集成}"
python3 e2e/scripts/package.py $M "$N" 2>&1 | tail -1
grep -n "^N_MODELS\|^EPOCHS, BATCH, LR, SEED\|^WIN_GROUPS\|^E04_EXTRA\|^E04_MTL\|^TRAIN_START\|^TARGET" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-120
rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert|截面|members|models" | tail -8
