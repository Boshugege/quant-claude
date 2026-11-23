#!/bin/bash
# R40（10-26，信息方向）：树成员从 G4 → G6 → G8 每加一批端到端通道都明显变强（树单独 2023 IC 0.0962 → 0.1004 → 0.1048），
# 说明 book2／book3／pos／state 里有信息；这几组只在旧架构（S7／S8）上给 Transformer 试过。E04 配方重测：
# R40a = R22a 配方 + book2,book3,pos,state（G6 的额外通道，不含 R32 已否决的 auc）。种子 42，fold-A，对照 R22a（同在 macstudio）。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log; R=e2e/runs
while pgrep -f "e2e/scripts/train_local.py|gbdt_member.py|verify_package.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R40 开始（E04 配方加 book2/book3/pos/state 通道）" >> $LIVE
echo "--run R40a_ch4 --arch e04 --norm rel2 --train 2020-01-01 2022-12-31 --val 2023-01-01 2023-12-31 --n-models 1 --zero-0930 --out-neutral 1.0 --skip-0930-loss --fuse --target cs_rank --win-groups shape,book,book2,book3,pos,state,tod --aux-w 0.5 --aux-n 2 --weight-ema 0.995 --mtl-w 0.5 --seed 42 --epochs 6 --train-neutral proxy_ind" > $R/queue_R40.txt
bash e2e/scripts/queue.sh $R/queue_R40.txt 1
r=R40a_ch4
[ -f $R/$r/transformer_model.json ] && [ ! -f $R/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py $R/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
echo "==================== $(date "+%m-%d %T") 两折对比 R40" >> $LIVE
TAG=R40 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py R22a=$R/R22a_ep6_mtl05 R40a=$R/$r 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R40 完成" >> $LIVE
