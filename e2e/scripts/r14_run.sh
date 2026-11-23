#!/bin/bash
# R14（10-19 夜）：5 轮的降换手是否稳定（换种子 43）、5 轮 + 日线 40 天、6 轮。训完补 2024 预测，两年对比 P13 三个种子与 R13e。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R14.txt 1
RUNS="R14a_P13_ep5_s43 R14b_P13_ep5_daily40 R14c_P13_ep6"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] || continue
  [ -f e2e/runs/$r/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R14" >> $LIVE
A="P13_s42=e2e/runs/R3f_rank_ep4 P13_s43=e2e/runs/R8b_rank_ep4_s43 P13_s44=e2e/runs/R12a_rank_ep4_s44 ep5_s42=e2e/runs/R13e_P13_ep5"; for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R14 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R14 完成" >> $LIVE
