#!/bin/bash
# R24（10-21 早）：多任务辅助头（R22a 两年 IC +0.002、换手 −0.005）的组合与确认——a 换种子 44，b 权重 1.0，c 换种子 43（R23：3 层专家换种子后中性，不再组合）。6 轮底座、种子 42，两年对比（含高、低离散半）。
#   对照：R21c（3 层种子 42）与 6 轮底座三种子
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
until grep -q "R23 完成" $LIVE; do sleep 30; done
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R24.txt 1
RUNS="R24a_mtl05_s44 R24b_mtl10 R24c_mtl05_s43"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] && [ ! -f e2e/runs/$r/pred_2024.parquet ] && { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R24" >> $LIVE
A="ep6_s42=e2e/runs/R14c_P13_ep6 ep6_s43=e2e/runs/R15a_P13_ep6_s43 ep6_s44=e2e/runs/R15b_P13_ep6_s44 R21c=e2e/runs/R21c_ep6_short3 R22a=e2e/runs/R22a_ep6_mtl05"; for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R24 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
PYTHONWARNINGS=ignore $PY e2e/scripts/stress_halves.py $A 2>&1 | grep -v -i warn >> $LIVE
echo "==================== $(date "+%m-%d %T") R24 完成" >> $LIVE
