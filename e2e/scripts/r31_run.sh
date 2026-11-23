#!/bin/bash
# R31（10-23，研究，不是 E16）：R30（多周期目标加重）IC 损失过大；改试极小的换手惩罚（训练时相邻截面同一股票的得分之差），
# 需要按日组批（--day-batch 1，以前单独看略差，所以 a 是只按日组批的对照）。b 惩罚 0.01、c 0.03。种子 42，对照 R22a。
cd ~/Desktop/work/quant
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R31.txt 1
RUNS="R31b_turn01 R31a_daybatch R31c_turn03"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R31" >> $LIVE
A="R22a=e2e/runs/R22a_ep6_mtl05 R24c=e2e/runs/R24c_mtl05_s43 R24a=e2e/runs/R24a_mtl05_s44"
for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R31 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R31 完成" >> $LIVE
