#!/bin/bash
# R27（10-21 晚）：提分候选，在 R22a（6 轮 + 多任务头 0.5）上：
#   a 训练时加 7 个中间时刻截面（S19 在旧底座上 IC_IR +0.05～0.10、压力 +0.03～0.08、换手 +0.05；端到端榜更看重 IC_IR／压力，重测）
#   c 宽度 96（多任务监督下再试加宽）  d 7 轮（多任务头可能推迟最佳轮数）  b 3 个中间时刻
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R26 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R27.txt 1
RUNS="R27a_extra7 R27c_d96 R27d_ep7 R27b_extra3"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R27" >> $LIVE
A="ep6_s42=e2e/runs/R14c_P13_ep6 R22a=e2e/runs/R22a_ep6_mtl05 R24c=e2e/runs/R24c_mtl05_s43 R24a=e2e/runs/R24a_mtl05_s44"
for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R27 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R27 完成" >> $LIVE
