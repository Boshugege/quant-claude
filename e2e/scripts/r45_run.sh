#!/bin/bash
# R45x（10-27，fold-A）：腿换手 0.1 + 中间时刻，配对只在正式时刻之间（中间时刻截面单独成单元）。R42x 混在一起配成 15 分钟间隔，换手只降 0.009。
# 若换手回到 R37a 的水平（约 0.535）且保留中间时刻的 IC／IC_IR 增益，最终候选可以用混合集成。种子 42，对照 R22a、R27a、R37a、R42x。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log; R=e2e/runs
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R45 开始（腿换手 0.1 + 中间时刻，配对只在正式时刻之间）" >> $LIVE
echo "--run R45x_leg01_x7pair --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --pair-units --leg-turn 0.1 --extra-times 09:45,10:15,10:45,11:15,13:15,13:45,14:15" > $R/queue_R45.txt
bash e2e/scripts/queue.sh $R/queue_R45.txt 1
r=R45x_leg01_x7pair
[ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
TAG=R45 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py R22a=$R/R22a_ep6_mtl05 R27a=$R/R27a_extra7 R37a=$R/R37a_leg01 R42x=$R/R42x_leg01_x7 R45x=$R/$r 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R45 完成" >> $LIVE
