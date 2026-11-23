#!/bin/bash
# R15（10-20 晨）：6 轮换种子 43／44，8 轮。训完补 2024 预测，两年对比 P13 三个种子与 R13e。
PY=/opt/homebrew/Caskroom/miniforge/base/envs/quant-e2e/bin/python
LIVE=e2e/runs/live.log
while pgrep -f "e2e/scripts/train_local.py" >/dev/null; do sleep 30; done
bash e2e/scripts/queue.sh e2e/runs/queue_R15.txt 1
RUNS="R15a_P13_ep6_s43 R15b_P13_ep6_s44 R15c_P13_ep8"
for r in $RUNS; do
  [ -f e2e/runs/$r/transformer_model.json ] || continue
  [ -f e2e/runs/$r/pred_2024.parquet ] || { echo "==================== $(date "+%m-%d %T") 预测 2024：$r" >> $LIVE; PYTHONWARNINGS=ignore $PY e2e/scripts/pred_year.py e2e/runs/$r 2024 2>&1 | grep -v -i warn >> $LIVE; }
done
echo "==================== $(date "+%m-%d %T") 两折对比 R15" >> $LIVE
A="P13_s42=e2e/runs/R3f_rank_ep4 P13_s43=e2e/runs/R8b_rank_ep4_s43 P13_s44=e2e/runs/R12a_rank_ep4_s44 ep5_s42=e2e/runs/R13e_P13_ep5 ep5_s43=e2e/runs/R14a_P13_ep5_s43 ep6_s42=e2e/runs/R14c_P13_ep6"; for r in $RUNS; do [ -f e2e/runs/$r/pred_2024.parquet ] && A="$A ${r%%_*}=e2e/runs/$r"; done
TAG=R15 PYTHONWARNINGS=ignore $PY e2e/scripts/cmp_runs.py $A 2>&1 | grep -v -i warn | grep -v "^  " >> $LIVE
echo "==================== $(date "+%m-%d %T") R15 完成" >> $LIVE
