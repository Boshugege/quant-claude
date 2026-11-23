#!/bin/bash
# R42x（10-26 晚，fold-A）：腿换手损失 0.1 能否与中间时刻（extra7）一起用。E16／E17b 的混合集成里两个成员加了中间时刻（集成 IC +0.002、IC_IR +0.02～0.03），
# E18 四个成员都没加；配对训练单元在加了中间时刻后变成 15 分钟间隔的相邻截面，没测过。种子 42，对照 R27a（中间时刻、无腿换手）与 R37a（腿换手、无中间时刻）。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log; R=e2e/runs
until grep -q "E18 四成员训练完成" $LIVE; do sleep 60; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R42 开始（腿换手 0.1 + 中间时刻，fold-A）" >> $LIVE
echo "--run R42x_leg01_x7 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --train-neutral proxy_ind --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --pair-units --leg-turn 0.1 --extra-times 09:45,10:15,10:45,11:15,13:15,13:45,14:15" > $R/queue_R42.txt
bash e2e/scripts/queue.sh $R/queue_R42.txt 1
r=R42x_leg01_x7
[ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
echo "==================== $(date "+%m-%d %T") 两折对比 R42" >> $LIVE
TAG=R42 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py R22a=$R/R22a_ep6_mtl05 R27a=$R/R27a_extra7 R37a=$R/R37a_leg01 R42x=$R/$r 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R42 完成" >> $LIVE
