#!/bin/bash
# R22（10-20，结构探索第三轮）：多任务监督与更长的 1 分钟回看。6 轮底座、种子 42，两年对比（含高、低离散半）。
#   a 多任务辅助头（未来 30 分钟已实现波动、量能变化）权重 0.5；b 权重 0.2；c 1 分钟回看 64 → 128
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R19 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R22.txt 1
RUNS="R22a_ep6_mtl05 R22b_ep6_mtl02 R22c_ep6_seq128"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R22" >> $LIVE
A="ep6_s42=e2e/runs/R14c_P13_ep6 ep6_s43=e2e/runs/R15a_P13_ep6_s43 ep6_s44=e2e/runs/R15b_P13_ep6_s44"; for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R22 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R22 完成" >> $LIVE
