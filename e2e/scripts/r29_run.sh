#!/bin/bash
# R29（10-22 晨）：R27a（训练加 7 个中间时刻）2023、2024 两年 IC_IR、压力都超出三个种子的范围 → 种子 44 再复核（对照 R24a = 多任务头 0.5 种子 44）；
#   b 更密的中间时刻（14 个，种子 42，对照 R27a）
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R28 完成" $LIVE; do sleep 60; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R29.txt 1
RUNS="R29a_extra7_s44 R29b_extra14"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R29" >> $LIVE
A="R22a=e2e/runs/R22a_ep6_mtl05 R24c=e2e/runs/R24c_mtl05_s43 R24a=e2e/runs/R24a_mtl05_s44 R27a=e2e/runs/R27a_extra7"
for r in R28a_extra7_s43 $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R29 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R29 完成" >> $LIVE
