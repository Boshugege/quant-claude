#!/bin/bash
# R33（10-24）：14:30 标签 = 连续竞价漂移 + 收盘竞价跳动，端到端在竞价部分 IC 只有 0.030（因子 0.052），漂移部分 0.135。
# 加两个只在 14:30 有值的多任务辅助目标：au 竞价跳动、dr 漂移（rv、lv 保留），种子 42，对照 R22a（E16 配方）。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
echo "==================== $(date "+%m-%d %T") R33 开始（多任务辅助目标加 14:30 竞价跳动／漂移）" >> $LIVE
bash e2e/scripts/queue.sh e2e/runs/queue_R33.txt 1
RUNS="R33a_mtl_au R33b_mtl_audr R33c_mtl_dr"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R33" >> $LIVE
A="R22a=e2e/runs/R22a_ep6_mtl05 R24a=e2e/runs/R24a_mtl05_s44 R24c=e2e/runs/R24c_mtl05_s43"
for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R33 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R33 完成" >> $LIVE
