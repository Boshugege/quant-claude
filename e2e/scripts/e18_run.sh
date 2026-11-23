#!/bin/bash
# E18（10-26）：端到端榜为降换手付钱（同一快照 P17 0.86791 对 P16 0.84776，+0.020：换手 −0.036，IC −0.0015、IC_IR −0.030）。
# E16 配方 + 腿换手损失 0.1（= P17 配方，配对训练单元）全量 2020–24，四个成员种子 42–45（不加中间时刻：腿换手损失与中间时刻的组合没测过），
# 之后接上 E17b 已训好的树成员（G8 配置，与 Transformer 成员无关，不用重训）。约 6 小时。
cd ~/Desktop/work/quant
LIVE=e2e/runs/live.log; R=e2e/runs
while pgrep -f "e2e/scripts/train_local.py|gbdt_member.py|verify_package.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") E18 开始（E16 配方 + 腿换手损失 0.1，全量四成员）" >> $LIVE
: > $R/queue_E18.txt
for s in 42 43 44 45; do
  echo "--run F4$((s-42))_2020-24_E18_leg01_s$s --arch e04 --norm rel2 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --win-groups shape,book,tod --fuse --aux-w 0.5 --aux-n 2 --epochs 6 --n-models 1 --train 2020-01-01 2024-12-31 --val 2024-10-01 2024-12-31 --secs-per-batch 4 --no-analyze --weight-ema 0.995 --target cs_rank --mtl-w 0.5 --seed $s --pair-units --leg-turn 0.1" >> $R/queue_E18.txt
done
bash e2e/scripts/queue.sh $R/queue_E18.txt 1
echo "==================== $(date "+%m-%d %T") E18 四成员训练完成" >> $LIVE
