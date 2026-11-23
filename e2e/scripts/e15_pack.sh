#!/bin/bash
# E15 打包：取回四个成员（F20–F23，P13 配方：秩目标 + 4 轮）→ 合并 → 打包（日线锚权重由环境变量 DAILY_MIX 指定，默认 0）→ 同步到 macstudio → verify_package。本机项目根目录运行。
set -e
cd "$(dirname "$0")/../.."
RUNS="F20_2020-24_rank_ep4_fuse_aux05_ema_tod_s42 F21_2020-24_rank_ep4_fuse_aux05_ema_tod_s43 F22_2020-24_rank_ep4_fuse_aux05_ema_tod_s44 F23_2020-24_rank_ep4_fuse_aux05_ema_tod_s45"
for r in $RUNS; do
    mkdir -p e2e/runs/$r
    rsync -a macstudio:Desktop/work/quant/e2e/runs/$r/{transformer_model.json,config.json,metrics.json,log.txt} e2e/runs/$r/
done
M=e2e/runs/F20-23_2020-24_rank_ep4_fuse_aux05_ema_tod_4seeds
STATS_ATOL=${STATS_ATOL:-0.02} python3 e2e/scripts/merge_members.py $M $(for r in $RUNS; do echo e2e/runs/$r; done)
N="${1:-E15_秩目标四轮_融合头_辅助目标0.5_EMA_当前时段_2020-24_4种子集成}"
python3 e2e/scripts/package.py $M "$N" 2>&1 | tail -1
grep -n "^N_MODELS\|^EPOCHS, BATCH, LR, SEED\|^WIN_GROUPS\|^E04_WEIGHT_EMA\|^E04_DAILY_MIX\|^TRAIN_START" "e2e/提交/$N/Transformer_modelsave_train.py" | cut -c1-100
rsync -a "e2e/提交/$N" "e2e/提交/$N.zip" macstudio:Desktop/work/quant/e2e/提交/
bash e2e/scripts/remote.sh e2e/scripts/verify_package.py "e2e/提交/$N" 2>&1 | grep -E "校验|通过|失败|Error|assert|截面|members|models" | tail -6
