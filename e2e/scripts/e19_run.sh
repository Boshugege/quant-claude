#!/bin/bash
# E19（10-26）：E16 配方 + 腿换手损失 W（默认 0.3），全量 2020–24 四成员（种子 42–45），排在 E18 之后。
# 依据：fold-A 上 L3（0.3）+ 树成员 IC 0.1044／换手 0.466（2023），比 L1 + 树（0.1095／0.548）换手低 0.08；端到端榜换手价格高（P16→P17 +0.020）。
# P18／P19 探针出分后可能改权重（先杀掉再用新的 W 重开）。用法：W=0.3 bash e2e/scripts/e19_run.sh
cd ~/Desktop/work/quant
W=${W:-0.3}; TAG=$(echo $W | tr -d .)
LIVE=e2e/runs/live.log; R=e2e/runs
until grep -q "E18 四成员训练完成" $LIVE; do sleep 60; done
while pgrep -f "e2e/scripts/train_local.py|gbdt_member.py|verify_package.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") E19 开始（E16 配方 + 腿换手损失 $W，全量四成员）" >> $LIVE
: > $R/queue_E19.txt
for s in 42 43 44 45; do
  echo "--run F5$((s-42))_2020-24_E19_leg${TAG}_s$s --arch e04 --norm rel2 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --win-groups shape,book,tod --fuse --aux-w 0.5 --aux-n 2 --epochs 6 --n-models 1 --train 2020-01-01 2024-12-31 --val 2024-10-01 2024-12-31 --secs-per-batch 4 --no-analyze --weight-ema 0.995 --target cs_rank --mtl-w 0.5 --seed $s --pair-units --leg-turn $W" >> $R/queue_E19.txt
done
bash e2e/scripts/queue.sh $R/queue_E19.txt 1
echo "==================== $(date "+%m-%d %T") E19 四成员训练完成" >> $LIVE
