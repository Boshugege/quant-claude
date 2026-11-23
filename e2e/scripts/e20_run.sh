#!/bin/bash
# E20（10-27，用户同意与 P23 探针并行）：E18g 的配方改成混合集成。成员 0、1 = E18 已训好的 F40（种子 42）、F41（43）；
# 成员 2、3 新训：腿换手 0.1 + 中间时刻（配对只在正式时刻之间，R45x：同 IC／换手下 IC_IR +0.04、压力 +0.04），种子 44、45，全量 2020–24。
cd ~/Desktop/work/quant
LIVE=e2e/runs/live.log; R=e2e/runs
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") E20 开始（两个新成员：腿换手 0.1 + 中间时刻，全量）" >> $LIVE
: > $R/queue_E20.txt
for s in 44 45; do
  echo "--run F6$((s-42))_2020-24_E20_leg01_x7_s$s --arch e04 --norm rel2 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --win-groups shape,book,tod --fuse --aux-w 0.5 --aux-n 2 --epochs 6 --n-models 1 --train 2020-01-01 2024-12-31 --val 2024-10-01 2024-12-31 --secs-per-batch 4 --no-analyze --weight-ema 0.995 --target cs_rank --mtl-w 0.5 --seed $s --pair-units --leg-turn 0.1 --extra-times 09:45,10:15,10:45,11:15,13:15,13:45,14:15" >> $R/queue_E20.txt
done
bash e2e/scripts/queue.sh $R/queue_E20.txt 1
echo "==================== $(date "+%m-%d %T") E20 两个新成员训练完成" >> $LIVE
